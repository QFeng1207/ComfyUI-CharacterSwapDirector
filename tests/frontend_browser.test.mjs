import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const packageRoot = process.env.CSD_NODE_MODULES;
let chromium;
try { ({ chromium } = require("playwright")); }
catch (error) { if (!packageRoot) throw error; ({ chromium } = require(path.join(packageRoot, "playwright"))); }
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const assets = {
  video: [{ id: "v1", kind: "video", file: "source-one.mp4", name: "视频一", url: "/fixtures/video.mp4", info: { width: 640, height: 360, duration: 20, fps_num: 24, fps_den: 1, frames: 480, has_audio: true } }, { id: "v2", kind: "video", file: "source-two.mp4", name: "视频二", url: "/fixtures/video.mp4", info: { width: 360, height: 640, duration: 9, fps_num: 30, fps_den: 1, frames: 270, has_audio: true } }],
  image: [{ id: "i1", kind: "image", file: "front.png", name: "正面参考", url: "/fixtures/reference.svg" }, { id: "i2", kind: "image", file: "side.png", name: "侧面参考 <img onerror=alert(1)>", url: "/fixtures/reference.svg" }],
};

const harness = `<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#27313d;padding:15px"><script type="module">
import { app } from '/scripts/app.js';
await import('/js/director.js');
await window.extension.setup();
class Node {
  constructor() { this.id=10; this.size=[980,950]; this.widgets=[{name:'project_json',value:'{}'}]; this.graph={change(){window.changeCount++}}; }
  onNodeCreated(){window.createdOriginal=true}
  onConfigure(){window.configuredOriginal=true}
  onResize(){window.resizedOriginal=true}
  onSerialize(info){info.otherExtension='preserved'}
  onRemoved(){window.removedOriginal=true}
  addDOMWidget(name,type,el,options={}) { const host=document.createElement('div');host.className='dom-widget';host.append(el);document.querySelector('#mount').append(host);const widget={name,type,element:el,options};this.widgets.push(widget);return widget }
  setSize(size){this.size=size}
  computeSize(){return [980,950]}
  setDirtyCanvas(){}
}
window.changeCount=0;
const mount=document.createElement('div'); mount.id='mount';mount.style.width='1000px';mount.setAttribute('node-id','10');mount.setAttribute('node-type','CSD_MaterialPlanner');document.body.append(mount);
const hidden=document.createElement('textarea');hidden.placeholder='project_json';const hiddenHost=document.createElement('div');hiddenHost.className='dom-widget';hiddenHost.append(hidden);mount.append(hiddenHost);
await window.extension.beforeRegisterNodeDef(Node,{name:'CSD_MaterialPlanner'});window.node=new Node();window.node.onNodeCreated();
class SaveNode extends Node { onExecuted(message){window.executedOriginal=message} }
await window.extension.beforeRegisterNodeDef(SaveNode,{name:'CSD_SaveResults'});window.saveNode=new SaveNode();window.saveNode.id=20;window.saveNode.type='CSD_SaveResults';
window.ui=window.node.__csd;window.ready=true;
</script></body></html>`;

test("director DOM supports media, anchors, roles, segments, original or mute audio and workflow lifecycle", async () => {
  const jobs = new Map(), calls = [];
  let serial = 0;
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const json = data => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(data)); };
    if (url.pathname === "/") { res.setHeader("Content-Type", "text/html"); res.end(harness); return; }
    if (url.pathname === "/scripts/app.js") { res.setHeader("Content-Type", "application/javascript"); res.end("export const app={graph:{getNodeById(id){return [window.node,window.saveNode].find(node=>String(node?.id)===String(id))}},registerExtension(e){window.extension=e},async queuePrompt(...args){window.queued=args}};"); return; }
    if (url.pathname === "/scripts/api.js") { res.setHeader("Content-Type", "application/javascript"); res.end("const events=new EventTarget();export const api={fetchApi:(...args)=>fetch(...args),addEventListener:(...args)=>events.addEventListener(...args)};window.mockEvents=events;"); return; }
    if (url.pathname.startsWith("/js/")) {
      const file = path.join(repo, url.pathname.slice(1));
      res.setHeader("Content-Type", file.endsWith(".css") ? "text/css" : "application/javascript"); res.end(await fs.readFile(file)); return;
    }
    if (url.pathname === "/fixtures/reference.svg" || url.pathname === "/fixtures/mask.svg") {
      res.setHeader("Content-Type", "image/svg+xml"); res.end('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="black"/><rect x="200" y="40" width="150" height="290" fill="white"/></svg>'); return;
    }
    if (url.pathname.startsWith("/fixtures/")) { res.statusCode = 204; res.end(); return; }
    if (url.pathname === "/csd/assets") {
      const kind = url.searchParams.get("kind");
      json({ assets: kind === "video" ? assets.video.map(asset => ({ ...asset, info: undefined })) : assets[kind] }); return;
    }
    if (url.pathname === "/csd/asset") { json(assets[url.searchParams.get("kind")].find(asset => asset.file === url.searchParams.get("file"))); return; }
    if (url.pathname === "/csd/results") { json({results: []}); return; }
    if (url.pathname === "/csd/dependencies") { json({ dependencies: [{ name: "H3", available: true, message: "测试运行时可用" }] }); return; }
    if (["/csd/analyse", "/csd/track"].includes(url.pathname)) {
      let body = ""; for await (const chunk of req) body += chunk;
      const input = JSON.parse(body); calls.push({ path: url.pathname, body: input });
      const id = `job-${++serial}`;
      let result;
      if (url.pathname === "/csd/analyse") result = {
        shots: [{ id: "shot1", start: 0, end: 10, scene_id: "shot1", scene_start: 0, scene_end: 10 }, { id: "shot2", start: 10, end: 20, scene_id: "shot2", scene_start: 10, scene_end: 20 }],
        segments: [{ id: "seg_a", start: 0, end: 10, scene_id: "shot1", enabled: true, role_mode: "auto", role_ids: [], prompt: "" }, { id: "seg_b", start: 10, end: 20, scene_id: "shot2", enabled: true, role_mode: "auto", role_ids: [], prompt: "" }]
      };
      if (url.pathname === "/csd/track") result = { mask_frames: [{ time: 1, role_id: input.state.roles[0].id, url: "/fixtures/mask.svg" }] };
      jobs.set(id, { status: url.pathname === "/csd/track" && serial > 2 ? "running" : "complete", progress: 1, message: "测试完成", result });
      json({ job_id: id }); return;
    }
    if (url.pathname.startsWith("/csd/jobs/")) {
      const id = url.pathname.split("/")[3], job = jobs.get(id);
      if (url.pathname.endsWith("/cancel")) { Object.assign(job, { status: "cancelled", message: "已取消" }); calls.push({ path: url.pathname }); }
      json(job); return;
    }
    res.statusCode = 404; res.end();
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ ...(process.platform === "win32" ? { channel: "msedge" } : {}), headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}`); await page.waitForFunction(() => window.ready);
    await page.waitForTimeout(120);
    assert.equal(await page.evaluate(() => window.createdOriginal), true);
    assert.equal(await page.locator('textarea[placeholder="project_json"]').isVisible(), false);
    assert.equal(await page.evaluate(() => window.node.widgets[0].hidden), true);
    assert.equal(await page.evaluate(() => window.node.widgets.find(widget => widget.name === "director").serialize), false);
    assert.equal(await page.evaluate(() => window.node.widgets.find(widget => widget.name === "director").options.getMinHeight()), 1180);
    const load = async (kind, name) => {
      const section = page.locator(".csd-library-section").nth({ video: 0, image: 1 }[kind]);
      await section.getByRole("button", { name: "加载", exact: true }).click();
      await page.locator(".csd-dialog-list").getByRole("button", { name, exact: true }).click();
      await page.locator(".csd-dialog").getByRole("button", { name: "关闭", exact: true }).click();
    };
    await load("video", "source-one.mp4");
    await load("image", "front.png"); await load("image", "side.png");
    await page.getByRole("button", { name: "＋ 角色", exact: true }).click();
    const referenceCards = page.locator(".csd-image-choice");
    await referenceCards.nth(0).locator('input[type="checkbox"]').check();
    await referenceCards.nth(1).locator('input[type="checkbox"]').check();
    await referenceCards.nth(1).locator('input[type="radio"]').check();
    assert.equal(await page.evaluate(() => window.ui.role.primary_image_id), "i2");
    const box = await page.locator(".csd-preview canvas").boundingBox();
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3); await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.85); await page.mouse.up();
    const anchor = await page.evaluate(() => window.ui.video.bindings[0].anchors[0]);
    assert.equal(anchor.time, 0); assert.ok(anchor.box[0] >= 0 && anchor.box[3] <= 1);
    await page.evaluate(() => window.ui.seekTo(5.125));
    await page.getByRole("button", { name: "标识人物", exact: true }).click();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    assert.equal(await page.evaluate(() => window.ui.video.bindings[0].anchors.length), 2);
    assert.equal(await page.evaluate(() => window.ui.video.bindings[0].anchors[1].time), 5.125);
    const firstRole = await page.evaluate(() => window.ui.role.name);
    await page.getByLabel("原片目标人物特征",{exact:true}).fill("left red coat");
    await page.getByLabel("原片目标人物特征",{exact:true}).press("Tab");
    await page.getByLabel("人物模式",{exact:true}).selectOption("single");
    for (const [number,description] of [[2,"middle gray shirt"],[3,"right green dress"]]) {
      await page.getByRole("button",{name:"＋ 角色",exact:true}).click();
      assert.equal(await page.getByLabel("原片目标人物特征",{exact:true}).isVisible(),true);
      await page.locator('.csd-image-choice').first().locator('input[type="checkbox"]').check();
      await page.getByRole("button",{name:"绑定当前视频 / 框选",exact:true}).click();
      await page.getByLabel("原片目标人物特征",{exact:true}).fill(description);
      await page.getByLabel("原片目标人物特征",{exact:true}).press("Tab");
      const target=await page.locator('.csd-preview canvas').boundingBox();
      const x=.2+number*.15;
      await page.mouse.move(target.x+target.width*x,target.y+target.height*.3);await page.mouse.down();
      await page.mouse.move(target.x+target.width*(x+.1),target.y+target.height*.8);await page.mouse.up();
    }
    assert.equal(await page.getByLabel("人物模式",{exact:true}).inputValue(),"multi");
    const independent=await page.evaluate(()=>window.ui.video.bindings.map(b=>({role:b.role_id,description:b.description,anchors:b.anchors.length})));
    assert.equal(new Set(independent.map(b=>b.role)).size,3);
    assert.deepEqual(independent.map(b=>b.description),["left red coat","middle gray shirt","right green dress"]);
    assert.ok(independent.every(b=>b.anchors>0));
    await page.getByRole("button",{name:firstRole,exact:true}).click();
    assert.equal(await page.getByLabel("原片目标人物特征",{exact:true}).inputValue(),"left red coat");
    await page.getByLabel("MP预设",{exact:true}).selectOption("1");
    assert.deepEqual(await page.evaluate(()=>window.ui.state.settings.resolution),{enabled:true,width:1376,height:768});
    await page.getByLabel("宽(px)",{exact:true}).fill("512");await page.getByLabel("宽(px)",{exact:true}).press("Tab");
    await page.getByLabel("高(px)",{exact:true}).fill("512");await page.getByLabel("高(px)",{exact:true}).press("Tab");
    assert.equal(await page.getByLabel("百万像素",{exact:true}).inputValue(),"0.25");
    await page.getByRole("button",{name:"沿用默认尺寸",exact:true}).click();
    assert.equal(await page.evaluate(()=>window.ui.state.settings.resolution.enabled),false);
    await page.getByRole("button", { name: "按镜头自动分段", exact: true }).click();
    await page.waitForFunction(() => window.ui.video.segments.length === 2);
    await page.evaluate(() => window.ui.seekTo(5));
    await page.getByRole("button", { name: "在播放处分裂", exact: true }).click();
    assert.equal(await page.evaluate(() => window.ui.video.segments.length), 3);
    await page.getByRole("button", { name: "合并后一段", exact: true }).click();
    assert.equal(await page.evaluate(() => window.ui.video.segments.length), 2);
    const edge = await page.locator(".csd-boundary").boundingBox();
    await page.mouse.move(edge.x + 5, edge.y + 15); await page.mouse.down(); await page.mouse.move(edge.x + 65, edge.y + 15); await page.mouse.up();
    const segments = await page.evaluate(() => window.ui.video.segments);
    assert.equal(segments[0].end, segments[1].start); assert.notEqual(segments[0].end, 5);
    await page.getByRole("button", { name: "试跑所选段", exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.queued), [0, 1]);
    assert.equal(await page.evaluate(() => window.ui.state.settings.trial.enabled), true);
    await page.getByRole("button", { name: "恢复完整运行", exact: true }).click();
    const audioChoice = page.locator(".csd-settings select").nth(1);
    assert.deepEqual(await audioChoice.locator("option").evaluateAll(options => options.map(option => [option.value, option.textContent])), [["original", "保留原声"], ["mute", "静音"]]);
    assert.equal(await page.locator(".csd-library-section").count(), 2);
    assert.equal(await page.getByRole("button", { name: "＋ 声音", exact: true }).count(), 0);
    assert.equal(await page.locator(".csd-speakers").count(), 0);
    await audioChoice.selectOption("mute");
    assert.equal(await page.evaluate(() => window.ui.player.muted), true);
    assert.equal(await page.evaluate(() => JSON.parse(window.node.widgets[0].value).settings.audio_mode), "mute");
    await audioChoice.selectOption("original");
    assert.equal(await page.evaluate(() => window.ui.player.muted), false);
    assert.equal(await page.getByRole("button", { name: "跟踪 / 蒙版", exact: true }).count(), 0);
    await page.getByRole("button", { name: "加载画面参考", exact: true }).click();
    await page.locator(".csd-dialog-list").getByRole("button", { name: "side.png", exact: true }).click();
    await page.locator(".csd-dialog").getByRole("button", { name: "关闭", exact: true }).click();
    assert.equal(await page.evaluate(() => window.ui.state.frame_references[0].kind), "guide");
    assert.equal(await page.evaluate(() => window.ui.state.frame_references[0].video_id), "v1");
    assert.equal(await page.locator(".csd-frame-card").count(), 1);
    await load("video", "source-two.mp4");
    await page.getByRole("button", { name: "框选人物", exact: true }).click();
    const portrait = await page.locator(".csd-preview canvas").boundingBox();
    await page.mouse.move(portrait.x + portrait.width * 0.47, portrait.y + portrait.height * 0.1); await page.mouse.down();
    await page.mouse.move(portrait.x + portrait.width * 0.53, portrait.y + portrait.height * 0.9); await page.mouse.up();
    assert.equal(await page.evaluate(() => window.ui.state.videos.filter(v => v.bindings.length).length), 2);
    await page.locator(".csd-library-section").nth(1).locator(".csd-asset").nth(1).getByRole("button", { name: "×", exact: true }).click();
    assert.equal(await page.evaluate(() => window.ui.role.primary_image_id), "i1");
    await page.locator(".csd-role-editor>input").fill('<img src=x onerror="alert(1)">'); await page.locator(".csd-role-editor>input").press("Tab");
    assert.equal(await page.locator(".csd-role-tabs img").count(), 0);
    const serialized = await page.evaluate(() => { const out={widgets_values:[]};window.node.onSerialize(out);return out });
    assert.equal(serialized.otherExtension, "preserved"); assert.equal(JSON.parse(serialized.widgets_values[0]).videos.length, 2);
    const changeCount = await page.evaluate(() => window.changeCount);
    await page.evaluate(() => { const out={widgets_values:[]};window.node.onSerialize(out) });
    assert.equal(await page.evaluate(() => window.changeCount), changeCount);
    const immediatelyRestored = await page.evaluate(() => {
      const saved = window.node.widgets[0].value;
      window.node.widgets[0].value = "{}"; window.ui.reload();
      window.node.onConfigure({ widgets_values: [saved], widgets_values_named: { project_json: saved } });
      const snapshot = { widgets_values: [] }; window.node.onSerialize(snapshot);
      return { videos: window.ui.state.videos.length, serialized: snapshot.widgets_values[0], saved };
    });
    assert.equal(immediatelyRestored.videos, 2);
    assert.equal(immediatelyRestored.serialized, immediatelyRestored.saved);
    await page.waitForTimeout(50);
    assert.equal(await page.evaluate(() => window.configuredOriginal), true); assert.equal(await page.evaluate(() => window.ui.state.videos.length), 2);
    assert.deepEqual(await page.evaluate(()=>window.ui.state.videos[0].bindings.map(b=>b.description)),["left red coat","middle gray shirt","right green dress"]);
    await page.getByRole("button", { name: "检查依赖", exact: true }).click(); await page.waitForFunction(() => document.querySelector(".csd-dialog").textContent.includes("H3"));
    await page.locator(".csd-dialog").getByRole("button", { name: "关闭", exact: true }).click();
    await page.evaluate(() => window.saveNode.onExecuted({ csd_results: [{ name: "result.mov <img onerror=alert(1)>", path: "C:/output/result.mov", url: "/fixtures/result.mov", preview_url: "/fixtures/preview.mp4", report: { size: "640x360", source_duration: 20 } }] }));
    assert.equal(await page.locator(".csd-output video").count(), 1);
    assert.equal(await page.evaluate(() => window.saveNode.widgets.find(widget => widget.name === "csd_results_preview").serialize), false);
    assert.equal(await page.locator(".csd-output a").getAttribute("href"), `http://127.0.0.1:${server.address().port}/fixtures/result.mov`);
    assert.equal(await page.locator(".csd-output img").count(), 0);
    assert.equal(await page.evaluate(() => window.executedOriginal.csd_results.length), 1);
    const beforeSegments = await page.evaluate(() => JSON.stringify(window.ui.state.videos[0].segments));
    await page.evaluate(() => window.saveNode.onExecuted({ csd_results: [{ name: "镜头测试.mov", result_id: "shot_result", planner_node: "10", video_id: "v1",
      url: "/fixtures/result.mov", preview_url: "/fixtures/preview.mp4", report: { source: "source-one.mp4",
        source_shots: [{ id: "verified_shot", start: 0, end: 20 }], warnings: ["10.000–20.000s 未标记人物，沿用原片"] } }] }));
    assert.deepEqual(await page.evaluate(() => window.ui.state.videos[0].shots), [{ id: "verified_shot", start: 0, end: 20 }]);
    assert.equal(await page.evaluate(() => JSON.stringify(window.ui.state.videos[0].segments)), beforeSegments);
    assert.match(await page.locator(".csd-status").textContent(), /未标记人物，沿用原片/);
    await page.evaluate(() => window.extension.onNodeOutputsUpdated({ "20": { csd_results: [{ name: "历史结果", url: "/fixtures/result.mov", preview_url: "/fixtures/preview.mp4" }] } }));
    assert.equal(await page.locator(".csd-output strong").textContent(), "历史结果");
    assert.equal(await page.locator(".csd-output video").count(), 1);
    await page.evaluate(() => window.mockEvents.dispatchEvent(new CustomEvent("csd.progress", { detail: { node: 10, progress: 0.6, stage: "替换人物", message: "测试视频生成进度" } })));
    assert.equal(await page.evaluate(() => window.ui.jobs.get("execution").status), "running");
    assert.equal(await page.locator(".csd-job").filter({ hasText: "测试视频生成进度" }).getByRole("button", { name: "取消", exact: true }).count(), 0);
    await page.screenshot({ path: path.join(repo, "tests", "frontend-qa.png"), fullPage: true });
    await page.evaluate(() => window.saveNode.onRemoved());
    await page.evaluate(() => window.node.onRemoved()); assert.equal(await page.evaluate(() => window.removedOriginal), true);
    assert.equal(errors.length, 0, errors.join("\n"));
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
});

test("director distinguishes automatic shot participation from manual zero roles and previews interpolation", async () => {
  const result = {
    shots: [{ id: "shot1", start: 0, end: 10 }, { id: "shot2", start: 10, end: 20 }],
    segments: [{ id: "seg_a", start: 0, end: 10, enabled: true, role_mode: "auto", role_ids: [] }, { id: "seg_b", start: 10, end: 20, enabled: true, role_mode: "auto", role_ids: [] }]
  };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const json = value => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(value)); };
    if (url.pathname === "/") { res.setHeader("Content-Type", "text/html"); res.end(harness); return; }
    if (url.pathname === "/scripts/app.js") { res.setHeader("Content-Type", "application/javascript"); res.end("export const app={graph:{getNodeById(){return window.node}},registerExtension(e){window.extension=e}};"); return; }
    if (url.pathname === "/scripts/api.js") { res.setHeader("Content-Type", "application/javascript"); res.end("const events=new EventTarget();export const api={fetchApi:(...args)=>fetch(...args),addEventListener:(...args)=>events.addEventListener(...args)};"); return; }
    if (url.pathname.startsWith("/js/")) {
      const file = path.join(repo, url.pathname.slice(1)); res.setHeader("Content-Type", file.endsWith(".css") ? "text/css" : "application/javascript"); res.end(await fs.readFile(file)); return;
    }
    if (url.pathname === "/csd/assets") { json({ assets: [] }); return; }
    if (url.pathname === "/csd/results") { json({ results: [] }); return; }
    if (url.pathname === "/csd/analyse") { for await (const _ of req) {} json({ job_id: "analysis" }); return; }
    if (url.pathname === "/csd/jobs/analysis") { json({ status: "complete", progress: 1, result }); return; }
    if (url.pathname.startsWith("/fixtures/")) { res.statusCode = 204; res.end(); return; }
    res.statusCode = 404; res.end();
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ ...(process.platform === "win32" ? { channel: "msedge" } : {}), headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
  const errors = []; page.on("pageerror", error => errors.push(error.message));
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}`); await page.waitForFunction(() => window.ready);
    await page.evaluate(asset => {
      const ui = window.ui;
      ui.state.roles = [{ id: "r1", name: "一号", image_ids: [], primary_image_id: "" }, { id: "r2", name: "二号", image_ids: [], primary_image_id: "" }];
      ui.state.videos = [{ ...asset, shots: [], segments: [{ id: "original", start: 0, end: 20, enabled: true, role_mode: "auto", role_ids: [], prompt: "" }], bindings: [
        { role_id: "r1", anchors: [{ time: 2, box: [.1, .2, .4, .8], points: [{ x: .8, y: .4, label: 0 }], strokes: [] }, { time: 8, box: [.5, .3, .8, .9], points: [], strokes: [] }] },
        { role_id: "r2", anchors: [{ time: 12, box: [.6, .2, .9, .8], points: [], strokes: [] }] }
      ] }];
      ui.state.selected_video_id = asset.id; ui.state.selected_role_id = "r1"; ui.selectedSegment = "original"; ui.save(true);
      window.previewBoxes = []; window.previewDashes = []; window.previewLabels = [];
      const draw = ui.drawBox.bind(ui); ui.drawBox = (ctx, fit, box) => { window.previewBoxes.push([...box]); draw(ctx, fit, box); };
      const ctx = ui.overlay.getContext("2d"), dash = ctx.setLineDash.bind(ctx), text = ctx.fillText.bind(ctx);
      ctx.setLineDash = values => { window.previewDashes.push([...values]); dash(values); };
      ctx.fillText = (...args) => { window.previewLabels.push(args[0]); text(...args); };
      ui.seekTo(5);
    }, assets.video[0]);
    assert.match(await page.locator(".csd-role-summary").textContent(), /尚未分析镜头，本段已标记候选/);
    assert.equal(await page.locator(".csd-role-checks input").first().isDisabled(), true);
    assert.equal(await page.evaluate(() => window.previewBoxes.length), 0);
    await page.getByLabel("参与角色", { exact: true }).selectOption("manual");
    await page.locator(".csd-role-checks").getByRole("checkbox", { name: "一号", exact: true }).uncheck();
    await page.locator(".csd-role-checks").getByRole("checkbox", { name: "二号", exact: true }).uncheck();
    assert.deepEqual(await page.evaluate(() => ({ mode: window.ui.segment.role_mode, roles: window.ui.segment.role_ids })), { mode: "manual", roles: [] });
    assert.match(await page.locator(".csd-role-summary").textContent(), /无，沿用原片/);
    await page.evaluate(() => window.ui.reload());
    assert.equal(await page.getByLabel("参与角色", { exact: true }).inputValue(), "manual");
    assert.deepEqual(await page.evaluate(() => window.ui.segment.role_ids), []);

    await page.evaluate(() => {
      const ui = window.ui;
      ui.video.segments = [{ id: "left", start: 0, end: 7, enabled: true, role_mode: "manual", role_ids: ["r1"], prompt: "保留近景" },
        { id: "right", start: 7, end: 20, enabled: true, role_mode: "manual", role_ids: ["r2"], prompt: "保留转场" }];
      ui.selectedSegment = "left"; ui.save(true);
    });
    await page.getByRole("button", { name: "按镜头自动分段", exact: true }).click();
    await page.waitForFunction(() => window.ui.video.shots.length === 2);
    assert.deepEqual(await page.evaluate(() => window.ui.video.segments.map(segment => [segment.start, segment.end, segment.role_mode, segment.role_ids, segment.prompt])),
      [[0, 7, "manual", ["r1"], "保留近景"], [7, 10, "manual", ["r2"], "保留转场"], [10, 20, "manual", ["r2"], "保留转场"]]);
    await page.evaluate(() => { window.previewBoxes = []; window.previewDashes = []; window.previewLabels = []; window.ui.seekTo(5); });
    const preview = await page.evaluate(() => ({ boxes: window.previewBoxes, dashes: window.previewDashes, labels: window.previewLabels }));
    assert.equal(preview.boxes.length, 1);
    [.3, .25, .6, .85].forEach((value, index) => assert.ok(Math.abs(preview.boxes[0][index] - value) < 1e-10));
    assert.ok(preview.dashes.some(values => values.length > 0));
    assert.ok(preview.labels.some(label => label.includes("一号 · 同镜头插值")));
    assert.match(await page.locator(".csd-center").textContent(), /并非自动视觉追踪/);
    await page.evaluate(() => {
      window.previewLabels = [];
      window.ui.video.bindings[0].anchors.push({ time: 9.99, box: [.1, .2, .4, .8], points: [], strokes: [] });
      window.ui.seekTo(10);
      window.ui.video.bindings[0].anchors.pop();
    });
    assert.deepEqual(await page.evaluate(() => window.previewLabels), ["二号 · 同镜头插值"]);
    assert.equal(await page.evaluate(() => window.ui.video.bindings[0].anchors[0].points[0].label), 0);
    await page.evaluate(() => { window.ui.selectedSegment = "left"; window.ui.seekTo(2); window.ui.renderTimeline(); });
    await page.getByRole("button", { name: "删除此帧锚点", exact: true }).click();
    await page.evaluate(() => window.ui.seekTo(8));
    await page.getByRole("button", { name: "删除此帧锚点", exact: true }).click();
    assert.match(await page.locator(".csd-role-summary").textContent(), /无，沿用原片/);
    assert.equal(await page.evaluate(() => window.ui.video.bindings[0].anchors.length), 0);
    await page.locator(".csd-shot-settings summary").click();
    await page.getByLabel("一号 · 本镜头", { exact: true }).selectOption("visible");
    await page.getByLabel("一号 · 可见部位", { exact: true }).selectOption("hand_arm");
    await page.getByLabel("一号 · 局部位置", { exact: true }).fill("left foreground holding gun");
    await page.getByLabel("一号 · 局部位置", { exact: true }).press("Tab");
    await page.getByLabel("镜头 0.000 秒动作顺序", { exact: true }).fill("Close-up first, then hand-only; keep source order.");
    await page.getByLabel("镜头 0.000 秒动作顺序", { exact: true }).press("Tab");
    assert.equal(await page.evaluate(() => window.ui.video.bindings[0].anchors.length), 0);
    assert.match(await page.locator(".csd-role-summary").textContent(), /一号/);
    await page.evaluate(() => window.ui.reload());
    const shotSettings = await page.evaluate(() => window.ui.video.shot_overrides[0]);
    assert.equal(shotSettings.roles[0].presence, "visible");
    assert.equal(shotSettings.roles[0].body_part, "hand_arm");
    assert.equal(shotSettings.roles[0].description, "left foreground holding gun");
    assert.match(shotSettings.prompt, /Close-up first/);
    assert.equal(errors.length, 0, errors.join("\n"));
    await page.evaluate(() => window.node.onRemoved());
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
});
