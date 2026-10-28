export const uid = (prefix = "item") => `${prefix}_${crypto.randomUUID().replaceAll("-", "")}`;

export function emptyProject() {
  return { version: 1, videos: [], images: [], roles: [], frame_references: [], settings: {
    person_mode: "multi", audio_mode: "original", segment_mode: "auto",
    max_segment_seconds: 10, scene_threshold: 0.2,
    trial: { enabled: false, video_id: "", segment_id: "" },
    revision: { enabled: false, video_id: "", base_result_id: "", scope: "selected", segment_ids: [] },
    candidate: { enabled: false, video_id: "", segment_id: "", time: 0 },
    resolution: { enabled: false, width: 0, height: 0 },
  }, selected_video_id: "", selected_role_id: "" };
}

export function parseProject(value) {
  if (!value || value === "{}") return emptyProject();
  const parsed = typeof value === "string" ? JSON.parse(value) : structuredClone(value);
  if (parsed.version !== 1) throw new Error("此工作流的素材计划版本不受支持。");
  delete parsed.audios;
  for (const key of ["videos", "images", "roles"]) {
    if (!Array.isArray(parsed[key])) throw new Error(`素材计划缺少 ${key} 列表。`);
  }
  parsed.settings = { ...emptyProject().settings, ...parsed.settings };
  if (["replace", "reference-timbre"].includes(parsed.settings.audio_mode)) parsed.settings.audio_mode = "original";
  if (!["original", "mute"].includes(parsed.settings.audio_mode)) throw new Error("请选择保留原声或静音。");
  parsed.settings.trial = { ...emptyProject().settings.trial, ...parsed.settings.trial };
  parsed.settings.revision = { ...emptyProject().settings.revision, ...parsed.settings.revision };
  parsed.settings.candidate = { ...emptyProject().settings.candidate, ...parsed.settings.candidate };
  parsed.settings.resolution = { ...emptyProject().settings.resolution, ...parsed.settings.resolution };
  parsed.frame_references ||= [];
  if (!Array.isArray(parsed.frame_references)) throw new Error("画面参考必须是列表。");
  for (const video of parsed.videos) {
    video.segments ||= [];
    video.bindings ||= [];
    video.shots ||= [];
    video.shot_overrides ||= [];
    for (const segment of video.segments) {
      segment.role_ids ||= [];
      segment.role_mode = segmentRoleMode(segment);
      if (!["auto", "manual"].includes(segment.role_mode)) throw new Error("分段角色请选择自动按镜头或手动选择。");
    }
    delete video.speakers;
  }
  for (const role of parsed.roles) {
    delete role.voice_id;
    role.image_ids ||= [];
    if (!role.image_ids.includes(role.primary_image_id)) role.primary_image_id = role.image_ids[0] || "";
  }
  if (!parsed.videos.some(v => v.id === parsed.selected_video_id)) parsed.selected_video_id = parsed.videos[0]?.id || "";
  if (!parsed.roles.some(r => r.id === parsed.selected_role_id)) parsed.selected_role_id = parsed.roles[0]?.id || "";
  return parsed;
}

export function addAsset(state, asset) {
  const key = { video: "videos", image: "images" }[asset.kind];
  if (!key) throw new Error("该素材类型不受支持。");
  const old = state[key].find(item => item.file === asset.file);
  if (old) {
    Object.assign(old, asset, { id: old.id });
    if (asset.kind === "video") state.selected_video_id = old.id;
    return old;
  }
  const item = { ...asset };
  if (asset.kind === "video") {
    item.segments = asset.info?.duration > 0 ? [{ id: uid("seg"), start: 0, end: asset.info.duration, enabled: true, role_mode: "auto", role_ids: [], prompt: "" }] : [];
    item.bindings = [];
    item.shots = [];
    item.shot_overrides = [];
    state.selected_video_id = item.id;
  }
  state[key].push(item);
  return item;
}

export function removeAsset(state, kind, id) {
  const key = { video: "videos", image: "images" }[kind];
  if (!key) throw new Error("该素材类型不受支持。");
  state[key] = state[key].filter(item => item.id !== id);
  state.frame_references = (state.frame_references || []).filter(item => kind === "image" ? item.image_id !== id : item.video_id !== id);
  if (kind === "image") for (const role of state.roles) {
    role.image_ids = role.image_ids.filter(image => image !== id);
    if (role.primary_image_id === id) role.primary_image_id = role.image_ids[0] || "";
  }
  if (kind === "video") {
    if (state.selected_video_id === id) state.selected_video_id = state.videos[0]?.id || "";
    if (state.settings.trial.video_id === id) state.settings.trial = { enabled: false, video_id: "", segment_id: "" };
    for (const setting of ["revision", "candidate"]) if (state.settings[setting]?.video_id === id) state.settings[setting].enabled = false;
  }
}

export function removeRole(state, id) {
  state.roles = state.roles.filter(role => role.id !== id);
  for (const video of state.videos) {
    video.bindings = video.bindings.filter(binding => binding.role_id !== id);
    for (const segment of video.segments) segment.role_ids = segment.role_ids.filter(role => role !== id);
    for (const shot of video.shot_overrides || []) shot.roles = (shot.roles || []).filter(role => role.role_id !== id);
    video.partial_role_observations = (video.partial_role_observations || []).filter(role => role.role_id !== id);
  }
  if (state.selected_role_id === id) state.selected_role_id = state.roles[0]?.id || "";
}

export function toggleRoleImage(role, imageId, enabled) {
  if (enabled && !role.image_ids.includes(imageId)) role.image_ids.push(imageId);
  if (!enabled) role.image_ids = role.image_ids.filter(id => id !== imageId);
  if (!role.image_ids.includes(role.primary_image_id)) role.primary_image_id = role.image_ids[0] || "";
}

export function ensureBinding(video, roleId) {
  let binding = video.bindings.find(item => item.role_id === roleId);
  if (!binding) { binding = { role_id: roleId, anchors: [], padding: 16, feather: 4 }; video.bindings.push(binding); }
  return binding;
}

export function alignDimension(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 32 || number > 16384) throw new Error("分辨率宽高必须在 32–16384 像素之间");
  const scaled = number / 32, floor = Math.floor(scaled);
  const rounded = scaled - floor === .5 ? floor + (floor % 2) : Math.round(scaled);
  return Math.min(16384, Math.max(32, rounded * 32));
}

export function dimensionsForMP(width, height, megapixels) {
  const mp = Number(megapixels);
  if (!Number.isFinite(mp) || mp < .1 || mp > 16) throw new Error("百万像素请选择 0.1–16 MP");
  const scale = Math.sqrt(mp * 1024 * 1024 / (width * height));
  return [alignDimension(width * scale), alignDimension(height * scale)];
}

export function ensureAnchor(binding, time, fps = 30) {
  let anchor = binding.anchors.find(item => Math.abs(item.time - time) < 0.5 / fps);
  if (!anchor) {
    anchor = { time, points: [], strokes: [] };
    binding.anchors.push(anchor);
    binding.anchors.sort((a, b) => a.time - b.time);
  }
  return anchor;
}

export function segmentRoleMode(segment) {
  return segment.role_mode || (segment.role_ids?.length ? "manual" : "auto");
}

function validPoint(point) {
  return Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
}

function validBox(box) {
  return Array.isArray(box) && box.length === 4 && box.every(value => Number.isFinite(value) && value >= 0 && value <= 1) && box[0] < box[2] && box[1] < box[3];
}

export function positiveAnchor(anchor) {
  return validBox(anchor.box) || (anchor.points || []).some(point => point.label !== 0 && validPoint(point)) ||
    (anchor.strokes || []).some(stroke => stroke.label !== 0 && stroke.points?.some(validPoint));
}

export function sourceShotRange(shot) {
  return { start: shot.scene_start ?? shot.start, end: shot.scene_end ?? shot.end };
}

export function shotOverride(video, shot, create = false) {
  const { start, end } = sourceShotRange(shot);
  let override = (video.shot_overrides || []).find(item => Math.abs(item.start - start) < 1e-5 && Math.abs(item.end - end) < 1e-5);
  if (!override && create) {
    override = { start, end, prompt: "", roles: [] };
    video.shot_overrides ||= [];
    video.shot_overrides.push(override);
  }
  return override;
}

export function shotRoleOverride(video, shot, roleId, create = false) {
  const override = shotOverride(video, shot, create);
  let role = override?.roles?.find(item => item.role_id === roleId);
  if (!role && create) {
    role = { role_id: roleId, presence: "auto", body_part: "", description: "" };
    override.roles ||= [];
    override.roles.push(role);
  }
  return role;
}

export function segmentParticipation(video, segment) {
  const shots = (video.shots || []).filter(shot => shot.start < segment.end && shot.end > segment.start);
  const known = shots.length > 0;
  const manual = segmentRoleMode(segment) === "manual";
  const rows = (known ? shots : [segment]).map(shot => {
    const { start, end } = known ? sourceShotRange(shot) : segment;
    const role_ids = (video.bindings || []).filter(binding => {
      if (manual && !segment.role_ids.includes(binding.role_id)) return false;
      const override = known ? shotRoleOverride(video, shot, binding.role_id) : null;
      if (override?.presence === "absent") return false;
      return override?.presence === "visible" || (binding.anchors || []).some(anchor => start <= anchor.time && anchor.time < end && positiveAnchor(anchor));
    }).map(binding => binding.role_id);
    return { start: Math.max(segment.start, shot.start), end: Math.min(segment.end, shot.end), role_ids };
  });
  return { known, rows, role_ids: [...new Set(rows.flatMap(row => row.role_ids))] };
}

export function interpolatedBox(video, binding, time) {
  const shot = (video.shots || []).find(item => item.start <= time && time < item.end);
  if (!shot) return null;
  const start = shot.scene_start ?? shot.start, end = shot.scene_end ?? shot.end;
  const anchors = (binding.anchors || []).filter(anchor => start <= anchor.time && anchor.time < end && validBox(anchor.box)).sort((a, b) => a.time - b.time);
  if (!anchors.length) return null;
  const right = anchors.find(anchor => anchor.time >= time) || anchors.at(-1);
  const left = anchors.filter(anchor => anchor.time <= time).at(-1) || anchors[0];
  const fraction = left.time === right.time ? 0 : (time - left.time) / (right.time - left.time);
  return left.box.map((value, index) => value + fraction * (right.box[index] - value));
}

export function applyAnalysis(video, result) {
  const previous = video.segments || [], used = new Set();
  const segments = [];
  for (const detected of result.segments) {
    const boundaries = [...new Set([detected.start, detected.end, ...previous.flatMap(segment => [segment.start, segment.end])
      .filter(time => detected.start < time && time < detected.end)])].sort((a, b) => a - b);
    for (let index = 0; index < boundaries.length - 1; index++) {
      const start = boundaries[index], end = boundaries[index + 1];
      const old = previous.find(segment => segment.start <= start && segment.end >= end);
      const role_mode = old ? segmentRoleMode(old) : "auto";
      const preferred = old?.start === start ? old.id : detected.id;
      const id = used.has(preferred) ? uid("seg") : preferred;
      used.add(id);
      segments.push({ ...detected, id, start, end, enabled: old?.enabled ?? detected.enabled ?? true,
        role_mode, role_ids: role_mode === "manual" ? [...old.role_ids] : [], prompt: old?.prompt || detected.prompt || "" });
    }
  }
  video.segments = segments;
  video.shots = structuredClone(result.shots || []);
  if (result.info) video.info = result.info;
}

export function frameRect(rect, width, height) {
  if (!(width > 0 && height > 0 && rect.width > 0 && rect.height > 0)) return null;
  const scale = Math.min(rect.width / width, rect.height / height);
  const fittedWidth = width * scale, fittedHeight = height * scale;
  return { left: rect.left + (rect.width - fittedWidth) / 2, top: rect.top + (rect.height - fittedHeight) / 2, width: fittedWidth, height: fittedHeight };
}

export function normalizedPoint(clientX, clientY, rect, width, height, clamp = false) {
  const fitted = frameRect(rect, width, height);
  if (!fitted) return null;
  let x = (clientX - fitted.left) / fitted.width, y = (clientY - fitted.top) / fitted.height;
  if (!clamp && (x < 0 || x > 1 || y < 0 || y > 1)) return null;
  x = Math.max(0, Math.min(1, x)); y = Math.max(0, Math.min(1, y));
  return { x, y };
}

export function splitSegment(video, segmentId, time) {
  const index = video.segments.findIndex(segment => segment.id === segmentId);
  if (index < 0) throw new Error("请先选择一个分段。");
  const segment = video.segments[index];
  if (!(time > segment.start && time < segment.end)) throw new Error("播放位置需要在所选分段内部。");
  segment.role_mode = segmentRoleMode(segment);
  const second = { ...structuredClone(segment), id: uid("seg"), start: time };
  segment.end = time;
  video.segments.splice(index + 1, 0, second);
  return second;
}

export function mergeNext(video, segmentId) {
  const index = video.segments.findIndex(segment => segment.id === segmentId);
  if (index < 0 || index === video.segments.length - 1) throw new Error("所选分段后面没有可合并的分段。");
  const segment = video.segments[index], next = video.segments[index + 1];
  const manual = segmentRoleMode(segment) === "manual" || segmentRoleMode(next) === "manual";
  const roles = item => segmentRoleMode(item) === "manual" ? item.role_ids : segmentParticipation(video, item).role_ids;
  const role_ids = manual ? [...new Set([...roles(segment), ...roles(next)])] : [];
  segment.end = next.end;
  segment.role_mode = manual ? "manual" : "auto";
  segment.role_ids = role_ids;
  if (next.prompt && next.prompt !== segment.prompt) segment.prompt = [segment.prompt, next.prompt].filter(Boolean).join("\n");
  video.segments.splice(index + 1, 1);
  return segment;
}

export function moveBoundary(video, index, time) {
  if (index < 1 || index >= video.segments.length) return;
  const previous = video.segments[index - 1], next = video.segments[index];
  const fps = video.info?.fps_num / (video.info?.fps_den || 1) || 30;
  const margin = Math.min(1 / fps, (next.end - previous.start) / 4);
  const snapped = Math.round(time * fps) / fps;
  const bounded = Math.max(previous.start + margin, Math.min(next.end - margin, snapped));
  previous.end = bounded;
  next.start = bounded;
}

export function selectTrial(state, videoId, segmentId) {
  if (!state.videos.some(video => video.id === videoId && video.segments.some(segment => segment.id === segmentId))) throw new Error("请选择有效分段。");
  state.settings.trial = { enabled: true, video_id: videoId, segment_id: segmentId };
  state.settings.candidate.enabled = false;
}

export function addFrameReference(state, asset, context) {
  if (asset.kind !== "image" || !["scene", "guide"].includes(context.kind)) throw new Error("请选择图片与画面参考类型。");
  const video = state.videos.find(item => item.id === context.video_id);
  if (!video || !Number.isFinite(context.time) || context.time < 0 || context.time >= video.info?.duration) throw new Error("请选择原视频内的参考时间。");
  const image = addAsset(state, asset);
  const reference = { id: uid("frame"), image_id: image.id, video_id: video.id, time: context.time, kind: context.kind, enabled: true, purpose: context.purpose || "" };
  state.frame_references ||= [];
  state.frame_references.push(reference);
  return reference;
}
