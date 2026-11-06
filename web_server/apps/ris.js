/* FLUX 3 Klein front end.  Prompts are expanded ("enhanced") by the upsampler
   LLM in rampart-llamacpp (global promptEnhancer, loaded in web_server_conf.js), which
   batches concurrent requests; the caption is then rendered by RIS Engine
   (../../engine, python3r server.py), which steps several users' images together.

   URLs (relative to /apps/ris/):
     generate.json  POST {prompt, mode: preview|full, aspect, size, upsample, variety, seed} -> {id, seed}
                    preview: 4 images at ~256x256, from one caption or (variety) one caption each
     upscale.json   POST {id, index, size, prompt?} -> {id}; re-renders image `index` of job `id`
                    larger (prompt: a description, for images without a caption, e.g. uploads)
     edit.json      POST {id, index, prompt, size, seed?, box?} -> {id}; image `index` of job `id`
                    changed as the prompt says, at `size` MP in its shape; with box [x, y, w, h]
                    (image pixels) only that part changes, edited at `size` MP, and the result
                    is the whole image at its own size (margin?, pin?: see engine/server.py)
     upload.json    POST the image file itself (Content-Type image/*) ?name= -> the new
                    "upload" job, whose image can then be upscaled or edited
     status.json    ?id=  -> {state: enhancing|queued|running|done|error, ...}
     gallery.json   ?limit= -> {images: [...]}, newest first
     image.png      ?f=   -> a png from the image dir
     delete.json    POST {id}
     cancel.json    POST {id}; stops a job that is not done yet and deletes it
*/

rampart.globalize(rampart.utils);
var curl = require("rampart-curl");

var ENGINE = "http://127.0.0.1:" + (process.env.RIS_ENGINE_PORT || 8189);

var dataDir = (global.serverConf && serverConf.dataRoot)
    ? serverConf.dataRoot + "/ris"
    : process.scriptPath + "/../data/ris";
var jobDir = dataDir + "/jobs";
var imageDir = dataDir + "/images";   // the engine's --out
if (!stat(jobDir)) mkdir(jobDir);

var systemPrompt = JSON.parse(readFile(weightsDir + "/flux-3-image-klein-pup/system_prompts.json", true)).t2i;

var ASPECTS = {
    "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4], "3:2": [3, 2],
    "2:3": [2, 3], "16:9": [16, 9], "9:16": [9, 16]
};
var SIZES = { "1": 1, "2": 2, "4": 4 };   // megapixels

/* width and height near `mp` megapixels at the given aspect (or width/height ratio), multiples of 16 */
function dims(aspect, mp, ratio) {
    var a = ASPECTS[aspect] || ASPECTS["1:1"];
    var px = mp * 1024 * 1024;
    var w = Math.sqrt(px * (ratio || a[0] / a[1]));
    var h = px / w;
    var fit = Math.min(1, 4096 / Math.max(w, h));   // the engine's limits: 64-4096 px a side
    w = Math.max(64, w * fit);
    h = Math.max(64, h * fit);
    w = Math.round(w / 16) * 16;
    h = Math.round(h / 16) * 16;
    // rounding up can pass the engine's limit (4 MP at 3:2 would be 2512x1680)
    while (w * h > px) {
        if (w >= h) w -= 16;
        else h -= 16;
    }
    return [w, h];
}

/* a job's width/height ratio: uploads (and what is made from them) keep their own */
function jobRatio(job) {
    return job.ratio || (ASPECTS[job.aspect] ? ASPECTS[job.aspect][0] / ASPECTS[job.aspect][1] : job.width / job.height);
}

var PREVIEW_MP = 0.0625;   // 256x256 at 1:1
var PREVIEW_COUNT = 4;

function uuid4() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0;
        return (c === "x" ? r : (r & 3 | 8)).toString(16);
    });
}

function jobFile(id) {
    if (!/^[0-9a-f-]{36}$/.test(id)) return null;
    return jobDir + "/" + id + ".json";
}

function loadJob(id) {
    var f = jobFile(id);
    if (!f || !stat(f)) return null;
    return JSON.parse(readFile(f, true));
}

function saveJob(job) {
    writeFile(jobFile(job.id), JSON.stringify(job));
}

/* saveJob for a job that may have been canceled (its file deleted) meanwhile: not brought back */
function updateJob(job) {
    if (stat(jobFile(job.id))) saveJob(job);
}

/* send the job to the engine under the job's own id; returns an error string or undefined.
   Image i gets seed + i and captions[i] (one caption is used for every image).  A preview job
   renders PREVIEW_COUNT images, anything else one.  job.init: an image to start from, scaled to
   the job's size and noised to job.initT (upscales); job.ref: a reference image, resized to
   job.refMp megapixels first (edits). */
function submit(job, captions) {
    var n = job.kind === "preview" ? PREVIEW_COUNT : 1, images = [];
    for (var i = 0; i < n; i++)
        images.push({prompt: captions[Math.min(i, captions.length - 1)], seed: job.seed + i});
    var spec = {id: job.id, width: job.width, height: job.height, images: images};
    if (job.init) {
        spec.init = job.init;
        spec.init_t = job.initT;
        if (job.via) {
            spec.via = job.via.map(function (wh) { return {width: wh[0], height: wh[1]}; });
            spec.via_t = job.viaT;
        }
    }
    if (job.ref) {
        spec.refs = [job.ref];
        spec.ref_mp = job.refMp;
    }
    if (job.box) {
        spec.box = job.box;
        spec.box_mp = job.boxMp;
        if (job.margin != null) spec.margin = job.margin;
        if (job.pin === false) spec.pin = false;
    }
    var res = curl.fetch(ENGINE + "/generate", {postJSON: spec, maxTime: 10});
    var body = {};
    try { body = JSON.parse(res.text); } catch (e) {}
    if (res.status !== 200 || !body.id)
        return "The image engine rejected the job: " + (body.error || res.text || res.errMsg || "not running");
}

function fail(job, msg) {
    job.state = "error";
    job.error = msg;
    job.finished = Date.now();
    updateJob(job);
}

function generate(req) {
    var p = req.postData.content || {};
    var prompt = String(p.prompt || "").trim();
    if (!prompt) return {status: 400, json: {error: "Enter a prompt."}};

    var aspect = ASPECTS[p.aspect] ? p.aspect : "1:1";
    var kind = p.mode === "full" ? "full" : "preview";
    var mp = kind === "preview" ? PREVIEW_MP : (SIZES[p.size] || 1);
    var wh = dims(aspect, mp);
    var seed = parseInt(p.seed, 10);
    if (isNaN(seed) || seed < 0) seed = Math.floor(Math.random() * 281474976710656);

    var job = {id: uuid4(), kind: kind, prompt: prompt, aspect: aspect, size: mp, width: wh[0], height: wh[1],
               seed: seed, upsample: !!p.upsample, created: Date.now(), state: "queued"};
    var reply = {json: {id: job.id, seed: seed, width: wh[0], height: wh[1]}};

    if (!job.upsample) {
        var err = submit(job, [prompt]);
        if (err) return {status: 502, json: {error: err}};
        saveJob(job);
        return reply;
    }

    // Variety: one sampled caption per preview.  Otherwise one greedy caption, as in BFL's workflow.
    // The engine batches these with other users' requests and queues beyond nSeqMax.
    job.variety = kind === "preview" && p.variety !== false;
    var n = job.variety ? PREVIEW_COUNT : 1, captions = [], pending = n, failed = false;
    var sampling = job.variety
        ? {temp: 0.8, topK: 40, topP: 0.9, minP: 0.05}
        : {temp: 1.0, topK: 1, topP: 0.95, minP: 0};
    job.state = "enhancing";
    saveJob(job);
    // canceled (cancel.json deletes the job): checked a few times a second as tokens arrive, and then
    // every caption of the job is stopped, freeing the enhancer.  cancel.json may run on another
    // server thread, which cannot reach these handles, so the check is made here.
    var handles = [], checked = Date.now();
    function onToken() {
        if (failed || Date.now() - checked < 250) return;
        checked = Date.now();
        if (stat(jobFile(job.id))) return;
        failed = true;
        handles.forEach(function (h) { h.cancel(); });
    }
    for (var i = 0; i < n; i++) (function (i) {
        handles[i] = promptEnhancer.predictAsync(Object.assign({
            messages: [{role: "system", content: systemPrompt}, {role: "user", content: prompt}],
            maxTokens: 1200, repeatPenalty: 1.0, seed: (seed + i) % 4294967296, thinking: false
        }, sampling), onToken, function (r) {
            if (failed) return;
            if (!stat(jobFile(job.id))) { failed = true; return; }   // canceled meanwhile
            if (r.error) { failed = true; return fail(job, "Prompt enhancement failed: " + r.error); }
            captions[i] = r.fullText.trim();
            if (--pending) return;
            if (job.variety) job.captions = captions;
            else job.upsampled = captions[0];
            job.enhanced = Date.now();
            var err = submit(job, captions);
            if (err) return fail(job, err);
            // canceled while the engine was taking it: cancel it there too, and do not bring it back
            if (!stat(jobFile(job.id))) return curl.fetch(ENGINE + "/cancel", {postJSON: {id: job.id}, maxTime: 10});
            job.state = "queued";
            saveJob(job);
        });
    })(i);
    return reply;
}

/* re-render one image of a finished job at a larger size, starting from it: the image is
   scaled up, noised and denoised from there.  Tested against using it as a reference image,
   which copied the small preview's garbled small objects into the result.  A preview goes to
   1 MP at PREVIEW_INIT_T (0.941 kept the layout and cleaned up small objects; 0.964 and up
   start to change the layout), and on through 1 MP to bigger sizes: straight from 256 px to
   2 or 4 MP came out blurred.  A 1 MP or bigger image needs less noise: UPSCALE_INIT_T.  An
   uploaded photo less again, UPLOAD_INIT_T, to stay close to it (faces held up to about 0.85;
   going through 1 MP did not help there), with the user's description as the caption. */
var PREVIEW_INIT_T = 0.941, UPSCALE_INIT_T = 0.85, UPLOAD_INIT_T = 0.8;
var GENERIC_CAPTION = "A high-quality, detailed photograph.";

/* the caption an image was made from, or "" when there is none (uploads, edits) */
function captionOf(job, index) {
    if (job.captions) return job.captions[index];
    if (job.upsampled) return job.upsampled;
    return job.kind === "preview" || job.kind === "full" ? job.prompt : "";
}

function upscale(req) {
    var p = req.postData.content || {};
    var parent = loadJob(p.id || "");
    if (!parent || parent.state !== "done") return {status: 404, json: {error: "Unknown image"}};
    var files = jobFiles(parent);
    var src = files[parseInt(p.index, 10) || 0];
    if (!src) return {status: 404, json: {error: "Unknown image"}};

    var mp = SIZES[p.size] || 1;
    var ratio = jobRatio(parent);
    var wh = dims(parent.aspect, mp, ratio);
    var index = files.indexOf(src);
    var description = String(p.prompt || "").trim();
    var caption = description || captionOf(parent, index) || GENERIC_CAPTION;
    var fromPreview = parent.kind === "preview";
    var job = {id: uuid4(), kind: "upscale", parent: parent.id, init: src,
               initT: fromPreview ? PREVIEW_INIT_T : parent.kind === "upload" ? UPLOAD_INIT_T : UPSCALE_INIT_T,
               prompt: description || parent.prompt || "", upsampled: caption, upsample: parent.upsample,
               aspect: parent.aspect, ratio: parent.ratio, size: mp, width: wh[0], height: wh[1],
               seed: (parent.seed || 0) + index, created: Date.now(), state: "queued"};
    if (fromPreview && mp > 1) {
        job.via = [dims(parent.aspect, 1, ratio)];
        job.viaT = UPSCALE_INIT_T;
    }
    var err = submit(job, [caption]);
    if (err) return {status: 502, json: {error: err}};
    saveJob(job);
    return {json: {id: job.id, seed: job.seed, width: wh[0], height: wh[1]}};
}

/* change image `index` of job `id` as the prompt says.  The prompt goes to the model as
   written: the prompt enhancer cannot see images here (rampart-llamacpp has no vision for
   generation), and BFL's edit prompts are short instructions anyway. */
function edit(req) {
    var p = req.postData.content || {};
    var parent = loadJob(p.id || "");
    if (!parent || parent.state !== "done") return {status: 404, json: {error: "Unknown image"}};
    var files = jobFiles(parent);
    var src = files[parseInt(p.index, 10) || 0];
    if (!src) return {status: 404, json: {error: "Unknown image"}};
    var prompt = String(p.prompt || "").trim();
    if (!prompt) return {status: 400, json: {error: "Describe the change first."}};

    var mp = SIZES[p.size] || 1;
    var ratio = jobRatio(parent);
    var wh = dims(parent.aspect, mp, ratio);
    var seed = parseInt(p.seed, 10);
    if (isNaN(seed) || seed < 0) seed = Math.floor(Math.random() * 281474976710656);
    // the image as reference, at its own size up to 1 MP (previews at 0.25 MP, which kept their framing best)
    var job = {id: uuid4(), kind: "edit", parent: parent.id, ref: src, refMp: Math.min(1, Math.max(0.25, parent.size || 1)),
               prompt: prompt, aspect: parent.aspect, ratio: parent.ratio, size: mp, width: wh[0], height: wh[1],
               seed: seed, created: Date.now(), state: "queued"};
    // a box: only that part of the image changes, and the result keeps the image's own size
    var box = Array.isArray(p.box) && p.box.length === 4 ? p.box.map(Number) : null;
    if (box && box.every(isFinite)) {
        job.box = box;
        job.boxMp = mp;
        job.size = parent.size;
        job.width = parent.width;
        job.height = parent.height;
        if (p.margin != null && isFinite(p.margin)) job.margin = Number(p.margin);
        if (p.pin === false) job.pin = false;
    }
    var err = submit(job, [prompt]);
    if (err) return {status: 502, json: {error: err}};
    saveJob(job);
    return {json: {id: job.id, seed: seed, width: job.width, height: job.height}};
}

var MAX_UPLOAD = 40 * 1024 * 1024;

/* an uploaded image: written to the image dir as upload_<uuid>; the engine checks it, applies its
   EXIF rotation, scales it to at most 4 MP and stores it as an img_NNNNN_.png */
function upload(req) {
    var body = req.body;
    if (!body || !body.length) return {status: 400, json: {error: "No image received."}};
    if (body.length > MAX_UPLOAD) return {status: 413, json: {error: "The image is too large (at most 40 MB)."}};
    if (!/^image\//.test(req.postData["Content-Type"] || "")) return {status: 415, json: {error: "That is not an image file."}};
    var name = "upload_" + uuid4(), path = imageDir + "/" + name;
    var fh = fopen(path, "w");
    fwrite(fh, body);
    fclose(fh);
    var res = curl.fetch(ENGINE + "/import", {postJSON: {name: name}, maxTime: 60});
    var r = {};
    try { r = JSON.parse(res.text); } catch (e) {}
    if (stat(path)) rmFile(path);
    if (res.status !== 200 || !r.file)
        return {status: res.status === 400 ? 400 : 502, json: {error: r.error || "The image engine is not answering."}};
    var now = Date.now();
    var job = {id: uuid4(), kind: "upload", prompt: "", filename: String(req.query.name || "").slice(0, 200),
               files: [r.file], width: r.width, height: r.height, size: r.width * r.height / 1048576,
               ratio: r.width / r.height, created: now, finished: now, state: "done"};
    saveJob(job);
    return {json: job};
}

/* records from before previews have a single `file` */
function jobFiles(job) {
    return job.files || (job.file ? [job.file] : []);
}

function status(req) {
    var job = loadJob(req.query.id || "");
    if (!job) return {status: 404, json: {error: "Unknown job"}};
    if (job.state === "done" || job.state === "error" || job.state === "enhancing") return {json: job};

    var res = curl.fetch(ENGINE + "/status?id=" + job.id, {maxTime: 10});
    if (res.status === 404) {
        fail(job, "The image engine restarted and lost this job; please generate again.");
        return {json: job};
    }
    if (res.status !== 200) return {status: 502, json: {error: "The image engine is not answering."}};
    var st = JSON.parse(res.text);
    if (st.state === "done") {
        job.state = "done";
        job.files = st.files;
        job.finished = Date.now();
        updateJob(job);
        return {json: job};
    }
    if (st.state === "error" || st.state === "canceled") {
        fail(job, st.error || "Generation failed");
        return {json: job};
    }
    return {json: {id: job.id, state: st.state, ahead: st.position || 0}};
}

function gallery(req) {
    var limit = parseInt(req.query.limit, 10) || 60;
    var jobs = [];
    readDir(jobDir).forEach(function (f) {
        if (!/\.json$/.test(f)) return;
        try {
            var j = JSON.parse(readFile(jobDir + "/" + f, true));
            if (j.state !== "done") return;
            j.files = jobFiles(j);
            jobs.push(j);
        } catch (e) {}
    });
    jobs.sort(function (a, b) { return b.created - a.created; });
    return {json: {images: jobs.slice(0, limit)}};
}

function image(req) {
    var f = req.query.f || "";
    if (!/^img_\d+_\.png$/.test(f) || !stat(imageDir + "/" + f))
        return {status: 404, txt: "Not found"};
    return {png: "@" + imageDir + "/" + f, headers: {"Cache-Control": "max-age=31536000, immutable"}};
}

function del(req) {
    var p = req.postData.content || {};
    var job = loadJob(p.id || "");
    if (!job) return {status: 404, json: {error: "Unknown job"}};
    jobFiles(job).forEach(function (f) {
        if (/^img_\d+_\.png$/.test(f) && stat(imageDir + "/" + f)) rmFile(imageDir + "/" + f);
    });
    rmFile(jobFile(job.id));
    return {json: {ok: true}};
}

/* stop a job that is not done: in the engine, or before it gets there (an enhancement still
   running sees the job gone and does not submit it); the job and any images it already finished
   (a preview set can be part done) are deleted */
function cancel(req) {
    var p = req.postData.content || {};
    var job = loadJob(p.id || "");
    if (!job) return {status: 404, json: {error: "Unknown job"}};
    if (job.state === "done") return {status: 409, json: {error: "Already done"}};
    rmFile(jobFile(job.id));
    // the engine may have the job even while it is marked enhancing (just submitted)
    curl.fetch(ENGINE + "/cancel", {postJSON: {id: job.id}, maxTime: 10});
    var res = curl.fetch(ENGINE + "/status?id=" + job.id, {maxTime: 10}), files = [];
    try { files = JSON.parse(res.text).files || []; } catch (e) {}
    files.forEach(function (f) {
        if (/^img_\d+_\.png$/.test(f) && stat(imageDir + "/" + f)) rmFile(imageDir + "/" + f);
    });
    return {json: {ok: true}};
}

function page(req) {
    return {html: "@" + serverConf.htmlRoot + "/studio/index.html"};
}

module.exports = {
    "/":              page,
    "/index.html":    page,
    "/generate.json": generate,
    "/upscale.json":  upscale,
    "/edit.json":     edit,
    "/upload.json":   upload,
    "/status.json":   status,
    "/gallery.json":  gallery,
    "/image.png":     image,
    "/delete.json":   del,
    "/cancel.json":   cancel
};
