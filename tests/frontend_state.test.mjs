import test from "node:test";
import assert from "node:assert/strict";
import { emptyProject, parseProject, addAsset, removeAsset, removeRole, toggleRoleImage, ensureBinding, ensureAnchor, frameRect, normalizedPoint, splitSegment, mergeNext, moveBoundary, selectTrial, addFrameReference } from "../js/state.mjs";
import { alignDimension, dimensionsForMP } from "../js/state.mjs";
import { segmentRoleMode, positiveAnchor, segmentParticipation, interpolatedBox, applyAnalysis, shotOverride, shotRoleOverride } from "../js/state.mjs";

test("explicit partial roles and shot actions persist without adding anchors or leaking across cuts", () => {
  const { state, video } = project();
  ensureBinding(video, "r1");
  video.shots = [{ id: "s1", start: 0, end: 10 }, { id: "s2", start: 10, end: 20 }];
  Object.assign(shotRoleOverride(video, video.shots[1], "r1", true), { presence: "visible", body_part: "hand_arm", description: "left foreground" });
  shotOverride(video, video.shots[1], true).prompt = "lift both feet, then exit";
  assert.deepEqual(segmentParticipation(video, video.segments[0]).rows.map(row => row.role_ids), [[], ["r1"]]);
  assert.equal(video.bindings[0].anchors.length, 0);
  const restored = parseProject(JSON.stringify(state)).videos[0];
  assert.equal(shotOverride(restored, restored.shots[1]).prompt, "lift both feet, then exit");
  shotRoleOverride(restored, restored.shots[1], "r1").presence = "absent";
  assert.deepEqual(segmentParticipation(restored, restored.segments[0]).role_ids, []);
  removeRole(state, "r1");
  assert.deepEqual(video.shot_overrides[0].roles, []);
});

function project() {
  const state = emptyProject();
  const video = addAsset(state, { id: "v1", kind: "video", file: "v1.mp4", name: "视频", info: { width: 1920, height: 1080, duration: 20, fps_num: 24, fps_den: 1 } });
  state.roles.push({ id: "r1", name: "主角", image_ids: [], primary_image_id: "" });
  state.selected_role_id = "r1";
  return { state, video, role: state.roles[0] };
}

test("project reload preserves bindings and selected trial", () => {
  const { state, video, role } = project();
  toggleRoleImage(role, "i1", true);
  ensureAnchor(ensureBinding(video, role.id), 3.5).box = [0.1, 0.2, 0.5, 0.9];
  selectTrial(state, video.id, video.segments[0].id);
  assert.deepEqual(parseProject(JSON.stringify(state)), state);
});

test("newer plan version is rejected without discarding existing data", () => {
  assert.throws(() => parseProject({ ...emptyProject(), version: 2 }), /版本/);
});

test("resolution controls match native MP calculation and preserve legacy defaults", () => {
  const { state } = project(); delete state.settings.resolution;
  assert.deepEqual(parseProject(state).settings.resolution, {enabled:false,width:0,height:0});
  assert.deepEqual(dimensionsForMP(1920,1080,1),[1376,768]);
  assert.equal(alignDimension(720),704); // match Python round-half-to-even
  assert.equal(alignDimension(1080),1088);
  assert.throws(()=>dimensionsForMP(1920,1080,NaN),/百万像素/);
});

test("one image supports independent scene and corrected-frame references without becoming role identity", () => {
  const { state, video, role } = project();
  const asset = { id: "guide", kind: "image", file: "corrected.png" };
  addFrameReference(state, asset, { video_id: video.id, time: 1, kind: "scene" });
  addFrameReference(state, asset, { video_id: video.id, time: 5, kind: "guide" });
  assert.equal(state.images.length, 1);
  assert.equal(state.frame_references.length, 2);
  assert.notEqual(state.frame_references[0].id, state.frame_references[1].id);
  assert.equal(role.image_ids.length, 0);
  assert.deepEqual(parseProject(JSON.stringify(state)).frame_references, state.frame_references);
  removeAsset(state, "image", asset.id);
  assert.equal(state.frame_references.length, 0);
});

test("frame references reject out-of-range times and removing a video clears its revision", () => {
  const { state, video } = project();
  const image = { id: "guide", kind: "image", file: "corrected.png" };
  for (const time of [-1, 20, NaN]) assert.throws(() => addFrameReference(state, image, { video_id: video.id, time, kind: "guide" }), /参考时间/);
  addFrameReference(state, image, { video_id: video.id, time: 19.9, kind: "guide" });
  state.settings.revision = { enabled: true, video_id: video.id, base_result_id: "version", scope: "selected", segment_ids: [video.segments[0].id] };
  removeAsset(state, "video", video.id);
  assert.equal(state.frame_references.length, 0);
  assert.equal(state.settings.revision.enabled, false);
});

test("reloading the same file preserves its identity and bindings", () => {
  const { state, video } = project();
  ensureBinding(video, "r1"); state.selected_video_id = "other";
  const loaded = addAsset(state, { id: "new_server_id", kind: "video", file: "v1.mp4", name: "新名字", info: video.info });
  assert.equal(loaded.id, "v1"); assert.equal(state.videos.length, 1); assert.equal(loaded.bindings.length, 1); assert.equal(state.selected_video_id, "v1");
});

test("deleting a primary image promotes a remaining reference", () => {
  const { state, role } = project();
  addAsset(state, { kind: "image", id: "i1", file: "a.png" }); addAsset(state, { kind: "image", id: "i2", file: "b.png" });
  toggleRoleImage(role, "i1", true); toggleRoleImage(role, "i2", true);
  removeAsset(state, "image", "i1");
  assert.equal(role.primary_image_id, "i2"); assert.deepEqual(role.image_ids, ["i2"]);
  toggleRoleImage(role, "i2", false); assert.equal(role.primary_image_id, "");
});

test("deleting a role or trial video cleans every dependent reference", () => {
  const { state, role, video } = project();
  ensureBinding(video, role.id); video.segments[0].role_ids = [role.id];
  removeRole(state, role.id); assert.equal(video.bindings.length, 0); assert.deepEqual(video.segments[0].role_ids, []);
  selectTrial(state, video.id, video.segments[0].id); removeAsset(state, "video", video.id);
  assert.equal(state.settings.trial.enabled, false); assert.equal(state.selected_video_id, "");
});

test("legacy voice settings migrate without losing images, anchors or segment prompts", () => {
  const { state, video, role } = project();
  state.settings.audio_mode = "replace";
  state.audios = [{ id: "missing_voice", file: "deleted.wav" }]; role.voice_id = "missing_voice";
  video.speakers = [{ id: "speaker", role_id: role.id, intervals: [] }];
  ensureAnchor(ensureBinding(video, role.id), 2).box = [0.1, 0.1, 0.7, 0.9];
  video.segments[0].prompt = "保留服装细节";
  const restored = parseProject(state);
  assert.equal(restored.settings.audio_mode, "original");
  assert.equal("audios" in restored, false); assert.equal("voice_id" in restored.roles[0], false);
  assert.equal("speakers" in restored.videos[0], false);
  assert.deepEqual(restored.videos[0].bindings, video.bindings);
  assert.deepEqual(restored.videos[0].segments, video.segments);
  assert.equal(state.settings.audio_mode, "replace");
  state.settings.audio_mode = "mute";
  assert.equal(parseProject(state).settings.audio_mode, "mute");
});

test("unsupported audio modes and voice reference uploads fail clearly", () => {
  const state = emptyProject();
  state.settings.audio_mode = "unknown";
  assert.throws(() => parseProject(state), /保留原声或静音/);
  assert.throws(() => addAsset(state, { kind: "audio", id: "voice", file: "voice.wav" }), /不受支持/);
});

test("one role can be independently anchored in multiple videos at arbitrary times", () => {
  const { state, video, role } = project();
  const second = addAsset(state, { id: "v2", kind: "video", file: "v2.mp4", info: video.info });
  const firstBinding = ensureBinding(video, role.id), secondBinding = ensureBinding(second, role.id);
  ensureAnchor(firstBinding, 9.2, 24).points.push({ x: 0.2, y: 0.4, label: 1 });
  ensureAnchor(secondBinding, 2.5, 24).box = [0.3, 0.1, 0.7, 0.9];
  assert.equal(firstBinding.anchors[0].time, 9.2); assert.equal(secondBinding.anchors[0].time, 2.5);
  assert.notEqual(firstBinding, secondBinding);
  assert.equal(ensureAnchor(firstBinding, 9.201, 24), firstBinding.anchors[0]);
});

test("letterbox clicks map to video coordinates and reject black bars", () => {
  const bounds = { left: 50, top: 100, width: 600, height: 400 };
  assert.deepEqual(frameRect(bounds, 1920, 1080), { left: 50, top: 131.25, width: 600, height: 337.5 });
  assert.deepEqual(normalizedPoint(350, 300, bounds, 1920, 1080), { x: 0.5, y: 0.5 });
  assert.equal(normalizedPoint(350, 110, bounds, 1920, 1080), null);
  assert.deepEqual(normalizedPoint(350, 110, bounds, 1920, 1080, true), { x: 0.5, y: 0 });
});

test("portrait and zoomed video coordinates use the visible full frame", () => {
  const bounds = { left: 20, top: 30, width: 400, height: 200 };
  assert.deepEqual(normalizedPoint(220, 130, bounds, 1080, 1920), { x: 0.5, y: 0.5 });
  assert.equal(normalizedPoint(30, 130, bounds, 1080, 1920), null);
});

test("split, move and merge maintain continuous coverage and custom prompts", () => {
  const { video } = project(); video.segments[0].prompt = "服装完整";
  const first = video.segments[0], second = splitSegment(video, first.id, 8);
  second.prompt = "检查遮挡"; second.role_mode = "manual"; second.role_ids = ["r2"];
  assert.equal(first.end, second.start); moveBoundary(video, 1, 10.01);
  assert.equal(first.end, 10); assert.equal(second.start, 10);
  moveBoundary(video, 1, 50); assert.ok(second.start < second.end); assert.equal(first.start, 0); assert.equal(second.end, 20);
  mergeNext(video, first.id); assert.equal(video.segments.length, 1); assert.equal(first.end, 20); assert.equal(first.prompt, "服装完整\n检查遮挡"); assert.deepEqual(first.role_ids, ["r2"]);
});

test("splitting outside a segment and selecting a missing trial fails clearly", () => {
  const { state, video } = project();
  assert.throws(() => splitSegment(video, video.segments[0].id, 0), /内部/);
  assert.throws(() => selectTrial(state, "missing", "missing"), /有效分段/);
});

test("legacy empty selections become automatic and explicit selections remain manual", () => {
  const { state, video } = project();
  const second = splitSegment(video, video.segments[0].id, 10);
  delete video.segments[0].role_mode; delete second.role_mode;
  second.role_ids = ["r1"];
  const restored = parseProject(state);
  assert.deepEqual(restored.videos[0].segments.map(segmentRoleMode), ["auto", "manual"]);
  assert.deepEqual(restored.videos[0].segments.map(segment => segment.role_ids), [[], ["r1"]]);
  second.role_mode = "all";
  assert.throws(() => parseProject(state), /自动按镜头或手动选择/);
});

test("only positive boxes, points and strokes establish a person's presence", () => {
  assert.equal(positiveAnchor({ points: [], strokes: [] }), false);
  assert.equal(positiveAnchor({ box: [.1, .2, .1, .9] }), false);
  assert.equal(positiveAnchor({ points: [{ x: .2, y: .4, label: 0 }], strokes: [{ label: 0, points: [{ x: .3, y: .5 }] }] }), false);
  assert.equal(positiveAnchor({ box: [.1, .2, .5, .9] }), true);
  assert.equal(positiveAnchor({ points: [{ x: .2, y: .4 }] }), true);
  assert.equal(positiveAnchor({ strokes: [{ label: 1, points: [{ x: .3, y: .5 }] }] }), true);
});

test("automatic participation separates alternating shots and shares anchors only within a known shot", () => {
  const { video } = project();
  video.shots = [{ id: "s1", start: 0, end: 4, scene_start: 0, scene_end: 4 }, { id: "s2", start: 4, end: 8, scene_start: 4, scene_end: 8 }];
  ensureAnchor(ensureBinding(video, "r1"), 3).box = [.1, .2, .5, .9];
  ensureAnchor(ensureBinding(video, "r2"), 5).points.push({ x: .7, y: .5, label: 1 });
  ensureAnchor(ensureBinding(video, "r3"), 6).strokes.push({ points: [{ x: .8, y: .6 }], label: 1 });
  ensureAnchor(ensureBinding(video, "negative"), 1).points.push({ x: .7, y: .5, label: 0 });
  const segment = { start: 0, end: 8, role_mode: "auto", role_ids: [] };
  assert.deepEqual(segmentParticipation(video, segment).rows.map(row => row.role_ids), [["r1"], ["r2", "r3"]]);
  assert.deepEqual(segmentParticipation(video, { ...segment, end: 2 }).role_ids, ["r1"]);
  assert.deepEqual(segmentParticipation(video, { ...segment, role_mode: "manual", role_ids: ["r1"] }).rows.map(row => row.role_ids), [["r1"], []]);
  video.shots = [];
  const unanalysed = segmentParticipation(video, { ...segment, end: 2 });
  assert.equal(unanalysed.known, false); assert.deepEqual(unanalysed.role_ids, []);
});

test("manual zero-role selections survive reload, role deletion, split and merge without becoming automatic", () => {
  const { state, video } = project();
  const segment = video.segments[0];
  segment.role_mode = "manual"; segment.role_ids = ["r1"];
  ensureAnchor(ensureBinding(video, "r1"), 1).box = [.1, .2, .5, .9];
  removeRole(state, "r1");
  assert.equal(segmentRoleMode(segment), "manual");
  assert.deepEqual(segmentParticipation(video, segment).role_ids, []);
  const second = splitSegment(video, segment.id, 10);
  assert.equal(second.role_mode, "manual"); assert.deepEqual(second.role_ids, []);
  mergeNext(video, segment.id);
  const restored = parseProject(JSON.stringify(state));
  assert.equal(restored.videos[0].segments[0].role_mode, "manual");
  assert.deepEqual(restored.videos[0].segments[0].role_ids, []);
});

test("merged automatic segments stay automatic and mixed selections retain already marked roles", () => {
  const { video } = project();
  const first = video.segments[0], second = splitSegment(video, first.id, 10);
  ensureAnchor(ensureBinding(video, "r1"), 2).box = [.1, .2, .5, .9];
  ensureAnchor(ensureBinding(video, "r2"), 12).box = [.6, .2, .9, .9];
  mergeNext(video, first.id);
  assert.equal(first.role_mode, "auto"); assert.deepEqual(first.role_ids, []);
  const last = splitSegment(video, first.id, 10);
  first.role_mode = "manual"; first.role_ids = ["r1"];
  mergeNext(video, first.id);
  assert.equal(first.role_mode, "manual"); assert.deepEqual(first.role_ids, ["r1", "r2"]);
  assert.equal(second.role_mode, "auto"); assert.equal(last.role_mode, "auto");
});

test("preview interpolates the full box and holds endpoints without carrying boxes across a cut", () => {
  const { video } = project();
  video.shots = [{ id: "s1", start: 0, end: 4 }, { id: "s2", start: 4, end: 8 }];
  const binding = ensureBinding(video, "r1");
  ensureAnchor(binding, 1).box = [.1, .2, .4, .8];
  ensureAnchor(binding, 3).box = [.5, .3, .8, .9];
  const midpoint = interpolatedBox(video, binding, 2);
  [.3, .25, .6, .85].forEach((value, index) => assert.ok(Math.abs(midpoint[index] - value) < 1e-10));
  assert.deepEqual(interpolatedBox(video, binding, .5), binding.anchors[0].box);
  assert.deepEqual(interpolatedBox(video, binding, 3.8), binding.anchors[1].box);
  assert.equal(interpolatedBox(video, binding, 4), null);
  assert.equal(interpolatedBox(video, binding, 8), null);
  video.shots = []; assert.equal(interpolatedBox(video, binding, 2), null);
});

test("new shot analysis retains original manual boundaries, zero-role ranges and segment prompts", () => {
  const { video } = project();
  const first = video.segments[0], second = splitSegment(video, first.id, 3);
  first.role_mode = "manual"; first.role_ids = ["r1"]; first.prompt = "近景";
  second.role_mode = "manual"; second.role_ids = []; second.prompt = "保留无人镜头"; second.enabled = false;
  const bindings = JSON.stringify(video.bindings);
  const shots = [{ id: "shot1", start: 0, end: 2 }, { id: "shot2", start: 2, end: 20 }];
  applyAnalysis(video, { shots, segments: [{ id: "new1", start: 0, end: 2, scene_id: "shot1" }, { id: "new2", start: 2, end: 20, scene_id: "shot2" }] });
  assert.deepEqual(video.segments.map(segment => [segment.start, segment.end, segment.role_mode, segment.role_ids, segment.prompt, segment.enabled]),
    [[0, 2, "manual", ["r1"], "近景", true], [2, 3, "manual", ["r1"], "近景", true], [3, 20, "manual", [], "保留无人镜头", false]]);
  assert.equal(video.segments[0].id, first.id); assert.equal(video.segments[2].id, second.id);
  assert.equal(new Set(video.segments.map(segment => segment.id)).size, video.segments.length);
  assert.deepEqual(video.shots, shots); assert.equal(JSON.stringify(video.bindings), bindings);
});
