import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { uid, emptyProject, parseProject, addAsset, removeAsset, removeRole, toggleRoleImage, ensureBinding, ensureAnchor, frameRect, normalizedPoint, splitSegment, mergeNext, moveBoundary, selectTrial, addFrameReference } from "./state.mjs";
import { USAGE_INTRO, USAGE_TOPICS } from "./help.mjs";
import { alignDimension, dimensionsForMP } from "./state.mjs";
import { segmentRoleMode, segmentParticipation, interpolatedBox, applyAnalysis, sourceShotRange, shotOverride, shotRoleOverride } from "./state.mjs";

const UI_HEIGHT = 1180;
const nodes = new Map();

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = String(text);
  return el;
}

function button(text, action, className = "", title = "") {
  const el = element("button", className, text);
  el.type = "button";
  if (title) el.title = title;
  el.addEventListener("click", action);
  return el;
}

function select(options, value, change) {
  const el = element("select");
  for (const [key, text] of options) {
    const option = element("option", "", text);
    option.value = key;
    el.append(option);
  }
  el.value = value || "";
  el.addEventListener("change", () => change(el.value));
  return el;
}

function numeric(value, change, { min = 0, max, step = 0.01 } = {}) {
  const el = element("input");
  el.type = "number"; el.value = value; el.min = min; el.step = step;
  if (max !== undefined) el.max = max;
  el.addEventListener("change", () => {
    const n = Number(el.value);
    if (Number.isFinite(n)) change(n);
  });
  return el;
}

function check(text, checked, change, radio = false) {
  const label = element("label");
  const input = element("input"); input.type = radio ? "radio" : "checkbox"; input.checked = checked;
  input.addEventListener("change", () => change(input.checked));
  label.append(input, element("span", "", text));
  return label;
}

function field(text, input) {
  const label = element("label", "csd-field");
  input.setAttribute("aria-label", text);
  label.append(element("span", "", text), input);
  return label;
}

function mediaURL(url) {
  if (!url) return "";
  const parsed = new URL(url, location.href);
  if (parsed.origin !== location.origin || !["http:", "https:"].includes(parsed.protocol)) throw new Error("素材地址必须来自当前 ComfyUI 服务。");
  return parsed.href;
}

function timeText(time) {
  const safe = Math.max(0, Number(time) || 0);
  return `${Math.floor(safe / 60).toString().padStart(2, "0")}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

function scrollDirector(event) {
  const root = event.target?.closest?.(".csd");
  if (!root) return;
  event.preventDefault(); event.stopImmediatePropagation();
  const unit = event.deltaMode === 1 ? 18 : event.deltaMode === 2 ? root.clientHeight : 1;
  const dx = (event.shiftKey ? event.deltaY : event.deltaX) * unit, dy = (event.shiftKey ? 0 : event.deltaY) * unit;
  let target = event.target;
  while (target && root.contains(target)) {
    const style = getComputedStyle(target);
    const vertical = /auto|scroll/.test(style.overflowY) && target.scrollHeight > target.clientHeight;
    const horizontal = /auto|scroll/.test(style.overflowX) && target.scrollWidth > target.clientWidth;
    if ((vertical && dy) || (horizontal && dx)) {
      const top = target.scrollTop, left = target.scrollLeft;
      if (vertical) target.scrollTop += dy;
      if (horizontal) target.scrollLeft += dx;
      if (target.scrollTop !== top || target.scrollLeft !== left) return;
    }
    if (target === root) break;
    target = target.parentElement;
  }
}

async function request(path, options = {}) {
  const response = await api.fetchApi(path, options);
  const body = await response.text();
  let result;
  try { result = JSON.parse(body); }
  catch {
    if (!response.ok) throw new Error(body.trim() || `本地服务返回 HTTP ${response.status}`);
    throw new Error("本地服务返回了无效的 JSON，请查看 ComfyUI 日志。");
  }
  if (!response.ok) throw new Error(result.error || result.message || `本地服务返回 HTTP ${response.status}`);
  return result;
}

export class CharacterSwapDirector {
  constructor(node, root, widget) {
    this.node = node; this.root = root; this.widget = widget;
    this.state = emptyProject(); this.time = 0; this.selectedSegment = ""; this.tool = "box";
    this.jobs = new Map(); this.maskFrames = new Map(); this.maskImage = null; this.maskKey = "";
    this.showMask = true; this.zoom = 1; this.disposed = false; this.abort = new AbortController();
    this.urls = new Map(); this.assetList = { video: [], image: [] };
    this.history = []; this.referenceKind = "guide";
    this.build();
    this.reload();
    this.resolveAssets();
  }

  get video() { return this.state.videos.find(video => video.id === this.state.selected_video_id); }
  get role() { return this.state.roles.find(role => role.id === this.state.selected_role_id); }
  get segment() { return this.video?.segments.find(segment => segment.id === this.selectedSegment); }
  get fps() { return this.video?.info?.fps_num / (this.video?.info?.fps_den || 1) || 30; }

  build() {
    this.root.className = "csd";
    this.root.addEventListener("pointerdown", event => event.stopPropagation());
    this.root.addEventListener("keydown", event => event.stopPropagation());
    this.root.addEventListener("wheel", event => event.stopPropagation(), { passive: true });
    this.root.addEventListener("dragover", event => { if (event.dataTransfer?.types.includes("Files")) { event.preventDefault(); event.stopPropagation(); } });
    this.root.addEventListener("drop", event => {
      if (!event.dataTransfer?.files.length) return;
      event.preventDefault(); event.stopPropagation(); this.uploadFiles(event.dataTransfer.files);
    });

    const head = element("div", "csd-head"), brand = element("div");
    brand.append(element("div", "csd-kicker", "CHARACTER SWAP DIRECTOR"), element("div", "csd-brand", "人物替换 · 素材导演"));
    const actions = element("div", "csd-actions");
    for (const [kind, label] of [["video", "＋ 视频"], ["image", "＋ 图片"]]) {
      actions.append(button(label, () => this.pickFiles(kind)));
    }
    actions.append(button("使用说明", () => this.openHelp(), "", "查看使用说明"), button("检查依赖", () => this.inspectDependencies()));
    head.append(brand, actions);
    this.settings = element("div", "csd-settings");
    const workspace = element("div", "csd-workspace");
    this.library = element("div", "csd-library");
    const center = element("div", "csd-center");
    this.previewTitle = element("div", "csd-section-title", "原视频 · 框选需要替换的人物");
    this.preview = element("div", "csd-preview");
    this.player = element("video"); this.player.preload = "metadata"; this.player.playsInline = true;
    this.player.addEventListener("loadedmetadata", () => {
      if (this.player.currentSrc !== this.playerURL) return;
      this.playerReady = true; this.player.currentTime = Math.min(this.time, this.player.duration || 0); this.drawOverlay();
    });
    this.player.addEventListener("timeupdate", () => {
      if (!this.playerReady || this.player.readyState < 1) return;
      this.time = this.player.currentTime; this.updateTime(); this.drawOverlay();
    });
    this.player.addEventListener("pause", () => { this.playButton.textContent = "▶ 播放"; });
    this.player.addEventListener("play", () => { this.playButton.textContent = "❚❚ 暂停"; });
    this.player.addEventListener("error", () => this.status("视频预览无法解码；请检查素材或本地预览转码。", true));
    this.overlay = element("canvas");
    this.overlay.addEventListener("pointerdown", event => this.anchorStart(event));
    this.overlay.addEventListener("pointermove", event => this.anchorMove(event));
    this.overlay.addEventListener("pointerup", event => this.anchorEnd(event));
    this.overlay.addEventListener("pointercancel", () => { this.drag = null; this.drawOverlay(); });
    this.overlay.addEventListener("contextmenu", event => event.preventDefault());
    this.emptyPreview = element("div", "csd-preview-empty", "上传或加载视频，在右侧添加角色后框选要替换的人物。");
    this.preview.append(this.player, this.overlay, this.emptyPreview);
    const transport = element("div", "csd-transport");
    this.playButton = button("▶ 播放", () => {
      if (!this.video) return;
      if (this.player.paused) this.player.play().catch(error => this.status(error.message, true)); else this.player.pause();
    });
    this.seek = element("input"); this.seek.type = "range"; this.seek.min = 0; this.seek.step = 0.001; this.seek.value = 0;
    this.seek.addEventListener("input", () => this.seekTo(Number(this.seek.value)));
    this.timeLabel = element("span", "csd-time", "00:00.00 / 00:00.00");
    transport.append(this.playButton, this.seek, this.timeLabel);
    this.previewTools = element("div", "csd-preview-tools");
    this.anchorList = element("div", "csd-anchor-list");
    center.append(this.previewTitle, this.preview, transport, this.previewTools, this.anchorList, element("div", "csd-note", "框选用于告诉 H3 替换谁。镜头分析后，虚线框按同镜头锚点插值或沿用端点，并非自动视觉追踪；切镜后需重新标记。"));
    this.rolePanel = element("div", "csd-role-panel");
    workspace.append(this.library, center, this.rolePanel);
    this.timelinePanel = element("div", "csd-timeline-panel");
    this.framePanel = element("div", "csd-frame-panel");
    this.jobPanel = element("div", "csd-jobs");
    this.statusBar = element("div", "csd-status"); this.statusMessage = element("span", "", "准备就绪");
    this.statusBar.append(this.statusMessage, element("span", "", "分段处理 · 输出沿用原片尺寸 / 时长 / 速度"));
    this.root.append(head, this.settings, workspace, this.timelinePanel, this.framePanel, this.jobPanel, this.statusBar);
    this.resizeObserver = new ResizeObserver(() => this.drawOverlay()); this.resizeObserver.observe(this.preview);
  }

  reload() {
    try { this.state = parseProject(this.widget?.value); }
    catch (error) { this.status(`无法读取工作流：${error.message}`, true); return; }
    this.selectedSegment = this.video?.segments[0]?.id || "";
    this.time = 0; this.render();
  }

  save(render = false) {
    if (this.widget) this.widget.value = JSON.stringify(this.state);
    this.node.setDirtyCanvas?.(true, true);
    this.node.graph?.change?.();
    if (render) this.render();
  }

  status(text, error = false) {
    this.statusMessage.textContent = text; this.statusBar.classList.toggle("error", error);
  }

  render() {
    this.renderSettings(); this.renderLibrary(); this.renderRoles(); this.renderPreviewTools();
    this.renderTimeline(); this.renderFramePanel(); this.renderJobs(); this.updatePreview();
  }

  renderSettings() {
    this.settings.replaceChildren();
    this.settings.append(field("人物模式", select([["single", "单人"], ["multi", "多人"]], this.state.settings.person_mode, value => {
      this.state.settings.person_mode = value; this.save();
      if (value === "single" && this.video?.bindings.length > 1) this.status("此视频已绑定多个人物；单人模式请在每段仅选择一个角色。", true);
    })), field("声音", select([["original", "保留原声"], ["mute", "静音"]], this.state.settings.audio_mode, value => {
      this.state.settings.audio_mode = value; this.save(); this.player.muted = value === "mute";
    })), field("分段", select([["auto", "按镜头自动"], ["manual", "手动分段"]], this.state.settings.segment_mode, value => { this.state.settings.segment_mode = value; this.save(); })),
    field("最长秒数", numeric(this.state.settings.max_segment_seconds, value => { this.state.settings.max_segment_seconds = Math.max(0.1, value); this.save(); }, { min: 0.1, step: 1 })),
    field("镜头阈值", numeric(this.state.settings.scene_threshold, value => { this.state.settings.scene_threshold = Math.max(0, Math.min(1, value)); this.save(); }, { max: 1, step: 0.05 })));
    const resolution = this.state.settings.resolution, info = this.video?.info;
    const width = resolution.enabled ? resolution.width : info?.width || 1344;
    const height = resolution.enabled ? resolution.height : info?.height || 768;
    const row = element("div", "csd-resolution-settings");
    const setDimensions = (w, h) => {
      try {
        Object.assign(resolution, { enabled: true, width: alignDimension(w), height: alignDimension(h) });
        this.save(); this.renderSettings();
      } catch (error) { this.status(error.message, true); this.renderSettings(); }
    };
    row.append(check("自定义分辨率", resolution.enabled, enabled => {
      if (enabled) setDimensions(width, height);
      else { resolution.enabled = false; this.save(); this.renderSettings(); }
    }), field("宽(px)", numeric(width, value => setDimensions(value, height), { min: 32, max: 16384, step: 32 })),
      field("高(px)", numeric(height, value => setDimensions(width, value), { min: 32, max: 16384, step: 32 })));
    const setMP = value => {
      try { setDimensions(...dimensionsForMP(width, height, value)); }
      catch (error) { this.status(error.message, true); }
    };
    row.append(field("百万像素", numeric(Number((width * height / 1024 ** 2).toFixed(3)), setMP, { min: .1, max: 16, step: .1 })),
      field("MP预设", select([["", "选择"], ...[.1,.25,.5,.75,1,1.5,2,3,4,8,16].map(value => [String(value), `${value} MP`])], "", value => { if (value) setMP(value); })),
      button("沿用默认尺寸", () => { Object.assign(resolution, { enabled: false, width: 0, height: 0 }); this.save(); this.renderSettings(); }));
    row.append(element("span", "csd-note", resolution.enabled ? `生成 / 输出 ${width} × ${height} · 32像素对齐；比例变化时只生成新增背景，并保留换人后的中心画面。` : "未设置时沿用当前生成参数，输出保持各原片尺寸。修改宽高或MP即可启用自定义。"));
    this.settings.append(row);
  }

  assetURL(asset) {
    const resolved = this.urls.get(asset.file);
    return mediaURL(resolved?.proxy_url || resolved?.url);
  }

  renderLibrary() {
    this.library.replaceChildren();
    for (const [kind, key, label] of [["video", "videos", "参考视频"], ["image", "images", "角色图片"]]) {
      const assets = this.state[key].filter(item => kind !== "image" || !this.state.frame_references.some(ref => ref.image_id === item.id) || this.state.roles.some(role => role.image_ids.includes(item.id)));
      const section = element("div", "csd-library-section"), title = element("div", "csd-section-title", `${label} · ${assets.length}`);
      title.append(button("加载", () => this.openAssetPicker(kind))); section.append(title);
      if (!assets.length) section.append(element("div", "muted", "支持上传、加载或拖入文件"));
      for (const asset of assets) {
        const card = element("div", `csd-asset${this.state.selected_video_id === asset.id ? " selected" : ""}`);
        card.append(button(asset.name || asset.file, () => {
          if (kind === "video") this.chooseVideo(asset.id);
          else if (kind === "image" && this.role) { toggleRoleImage(this.role, asset.id, true); this.save(); this.renderRoles(); }
          else this.status("请先在右侧选择或添加角色，再绑定参考素材。", true);
        }, "csd-asset-title", asset.file));
        card.append(button("×", () => {
          removeAsset(this.state, kind, asset.id); this.save(true); this.status("素材已从计划移除，相关绑定已解除；原文件保留。");
        }, "csd-remove danger", "从本计划移除素材"));
        if (kind === "video") card.append(element("div", "muted", `${(asset.info?.duration || 0).toFixed(2)}s · ${asset.info?.width || "?"}×${asset.info?.height || "?"}`));
        if (kind === "image") {
          const image = element("img"); image.alt = asset.name || "参考图"; image.loading = "lazy";
          try { image.src = this.assetURL(asset); } catch (error) { this.status(error.message, true); }
          card.append(image);
        }
        section.append(card);
      }
      this.library.append(section);
    }
  }

  chooseVideo(id) {
    this.player.pause(); this.state.selected_video_id = id; this.time = 0;
    this.selectedSegment = this.video?.segments[0]?.id || "";
    if (this.state.settings.revision.video_id !== id) this.state.settings.revision.enabled = false;
    this.save(true);
    this.loadHistory();
  }

  renderRoles() {
    this.rolePanel.replaceChildren();
    const title = element("div", "csd-section-title", "角色 / 人物绑定");
    title.append(button("＋ 角色", () => {
      const role = { id: uid("role"), name: `角色 ${this.state.roles.length + 1}`, image_ids: [], primary_image_id: "", description: "" };
      this.state.roles.push(role); this.state.selected_role_id = role.id; this.save(true);
    }));
    const tabs = element("div", "csd-role-tabs");
    for (const role of this.state.roles) tabs.append(button(role.name, () => {
      this.state.selected_role_id = role.id; this.save(); this.renderRoles(); this.renderPreviewTools(); this.drawOverlay();
    }, role.id === this.state.selected_role_id ? "active" : ""));
    this.rolePanel.append(title, tabs);
    const role = this.role;
    if (!role) { this.rolePanel.append(element("div", "muted", "添加一个角色，把参考图片和视频人物绑定在一起。")); return; }
    const editor = element("div", "csd-role-editor"), name = element("input"); name.value = role.name;
    name.addEventListener("change", () => { role.name = name.value || "未命名角色"; this.save(); this.renderRoles(); this.renderTimeline(); });
    const description = element("textarea"); description.value = role.description || ""; description.placeholder = "角色描述 / 外观补充";
    description.addEventListener("change", () => { role.description = description.value; this.save(); });
    editor.append(name, description, element("label", "", "参考图 · 勾选绑定，主图决定外观"));
    const images = element("div", "csd-image-choices");
    for (const image of this.state.images.filter(item => !(this.state.frame_references || []).some(ref => ref.image_id === item.id) || role.image_ids.includes(item.id))) {
      const bound = role.image_ids.includes(image.id), card = element("div", `csd-image-choice${bound ? " bound" : ""}`);
      const thumbnail = element("img"); thumbnail.alt = image.name; thumbnail.title = image.name;
      try { thumbnail.src = this.assetURL(image); } catch (error) { this.status(error.message, true); }
      const controls = element("div", "csd-choice-controls");
      controls.append(check("", bound, enabled => { toggleRoleImage(role, image.id, enabled); this.save(); this.renderRoles(); }));
      const primary = check("主图", role.primary_image_id === image.id, () => {
        toggleRoleImage(role, image.id, true); role.primary_image_id = image.id; this.save(); this.renderRoles();
      }, true);
      primary.querySelector("input").name = `csd-primary-${this.node.id}-${role.id}`;
      controls.append(primary); card.append(thumbnail, controls); images.append(card);
    }
    if (!this.state.images.length) images.append(element("div", "muted", "先上传图片"));
    editor.append(images);
    const video = this.video, binding = video?.bindings.find(item => item.role_id === role.id);
    editor.append(element("label", "", binding ? `当前视频 · ${binding.anchors.length} 个人物锚点` : "当前视频 · 尚未选中人物"));
    const identity = element("textarea"); identity.value = binding?.description || ""; identity.placeholder = "原片目标人物特征（多人时可补充：左侧、穿黑衣等）";
    identity.setAttribute("aria-label", "原片目标人物特征");
    identity.disabled = !video;
    identity.addEventListener("change", () => {
      const current = this.bindCurrentRole(); if (!current) return;
      current.description = identity.value; this.save(); this.renderRoles();
    });
    editor.append(identity);
    const bind = button(binding ? "框选此角色" : "绑定当前视频 / 框选", () => {
      if (!this.bindCurrentRole()) return;
      this.tool = "box"; this.player.pause(); this.save(true);
      this.status(`已选择 ${this.role.name}；请在原视频预览中拖框定位此人物。`);
    }); bind.disabled = !video; editor.append(bind);
    if (binding) {
      editor.append(button("解除当前视频绑定", () => {
        video.bindings = video.bindings.filter(item => item.role_id !== role.id);
        for (const segment of video.segments) segment.role_ids = segment.role_ids.filter(id => id !== role.id);
        for (const shot of video.shot_overrides || []) shot.roles = (shot.roles || []).filter(item => item.role_id !== role.id);
        this.invalidateMask(); this.save(true);
      }, "danger"));
    }
    const bindingsTitle = element("label", "", "此角色绑定的视频"); editor.append(bindingsTitle);
    for (const item of this.state.videos.filter(v => v.bindings.some(b => b.role_id === role.id))) {
      const row = element("div", "csd-role-binding");
      row.append(element("span", "", item.name), button("定位", () => this.chooseVideo(item.id))); editor.append(row);
    }
    editor.append(button("删除角色", () => { removeRole(this.state, role.id); this.invalidateMask(); this.save(true); }, "danger"));
    this.rolePanel.append(editor);
  }

  renderPreviewTools() {
    this.previewTools.replaceChildren();
    for (const [tool, label] of [["box", "框选人物"], ["point", "标识人物"]]) this.previewTools.append(button(label, () => { this.tool = tool; this.renderPreviewTools(); }, this.tool === tool ? "active" : ""));
    this.showMask = false;
    this.previewTools.append(button("原片截图作场景参考", () => this.exportFrame(null, this.time, true)));
    this.previewTools.append(button("删除此帧锚点", () => {
      const binding = this.video?.bindings.find(item => item.role_id === this.role?.id);
      if (!binding) return;
      binding.anchors = binding.anchors.filter(anchor => Math.abs(anchor.time - this.time) >= 0.5 / this.fps);
      this.invalidateMask(); this.save(); this.renderPreviewTools(); this.renderRoles(); this.renderTimeline(); this.drawOverlay();
    }, "danger"));
    this.anchorList.replaceChildren();
    const binding = this.video?.bindings.find(item => item.role_id === this.role?.id);
    for (const anchor of binding?.anchors || []) this.anchorList.append(button(`${timeText(anchor.time)}${anchor.box ? " ▢" : ""}${anchor.strokes?.length ? " ✎" : ""}`, () => this.seekTo(anchor.time), "", "定位此角色的人物锚点"));
  }

  updatePreview() {
    const video = this.video;
    this.emptyPreview.hidden = Boolean(video);
    this.previewTitle.textContent = video ? `${video.name} · ${this.role?.name || "请添加角色"}` : "视频预览 · 点选或框选人物";
    const url = video ? this.assetURL(video) : "";
    const fileKey = video ? `${video.file}:${url}` : "";
    if (this.previewFile !== fileKey) {
      this.playerReady = false; this.playerURL = url; this.player.pause(); this.previewFile = fileKey; this.maskKey = ""; this.maskImage = null;
      if (video && url) this.player.src = url;
      else { this.player.removeAttribute("src"); this.player.load(); }
    }
    this.player.muted = this.state.settings.audio_mode === "mute";
    this.seek.max = video?.info?.duration || 0; this.updateTime(); this.drawOverlay();
  }

  seekTo(time) {
    this.time = Math.max(0, Math.min(this.video?.info?.duration || 0, time));
    if (this.player.readyState >= 1) this.player.currentTime = this.time;
    this.updateTime(); this.drawOverlay();
  }

  updateTime() {
    this.seek.value = this.time; this.timeLabel.textContent = `${timeText(this.time)} / ${timeText(this.video?.info?.duration)}`;
    if (this.playhead && this.video?.info?.duration) this.playhead.style.left = `${this.time / this.video.info.duration * 100}%`;
  }

  videoPoint(event, clamp = false) {
    const info = this.video?.info;
    return normalizedPoint(event.clientX, event.clientY, this.overlay.getBoundingClientRect(), info?.width || this.player.videoWidth, info?.height || this.player.videoHeight, clamp);
  }

  bindCurrentRole() {
    if (!this.video || !this.role) return null;
    if (this.state.settings.person_mode === "single" && this.video.bindings.some(binding => binding.role_id !== this.role.id)) {
      this.state.settings.person_mode = "multi";
      this.renderSettings(); this.status("已切换多人模式；每个角色分别绑定原片人物。");
    }
    return ensureBinding(this.video, this.role.id);
  }

  anchorStart(event) {
    event.preventDefault();
    const point = this.videoPoint(event);
    if (!point) return;
    if (!this.role) { this.status("请先添加或选择一个角色。", true); return; }
    this.player.pause(); this.overlay.setPointerCapture(event.pointerId);
    const binding = this.bindCurrentRole(), anchor = ensureAnchor(binding, this.time, this.fps);
    const label = event.button === 2 || event.shiftKey || this.tool === "erase" ? 0 : 1;
    this.drag = { start: point, current: point, tool: this.tool, anchor, label, points: [point], pointerId: event.pointerId };
    this.drawOverlay();
  }

  anchorMove(event) {
    if (!this.drag) return;
    const point = this.videoPoint(event, true); if (!point) return;
    this.drag.current = point;
    if (["brush", "erase"].includes(this.drag.tool)) {
      const last = this.drag.points.at(-1);
      if (Math.hypot(point.x - last.x, point.y - last.y) > 0.002) this.drag.points.push(point);
    }
    this.drawOverlay();
  }

  anchorEnd(event) {
    if (!this.drag) return;
    const drag = this.drag; this.drag = null;
    if (this.overlay.hasPointerCapture(event.pointerId)) this.overlay.releasePointerCapture(event.pointerId);
    if (drag.tool === "box") {
      const x0 = Math.min(drag.start.x, drag.current.x), x1 = Math.max(drag.start.x, drag.current.x);
      const y0 = Math.min(drag.start.y, drag.current.y), y1 = Math.max(drag.start.y, drag.current.y);
      if (x1 - x0 < 0.002 || y1 - y0 < 0.002) drag.anchor.points.push({ ...drag.start, label: drag.label });
      else drag.anchor.box = [x0, y0, x1, y1];
    } else if (drag.tool === "point") drag.anchor.points.push({ ...drag.current, label: drag.label });
    else drag.anchor.strokes.push({ points: drag.points, radius: 0.015, label: drag.label });
    this.invalidateMask(); this.save(); this.renderRoles(); this.renderPreviewTools(); this.renderTimeline(); this.drawOverlay();
    this.status(`已绑定 ${this.role.name} · ${timeText(drag.anchor.time)}；在时间轴选择分段后可“试跑所选段”。`);
  }

  invalidateMask() {
    if (this.video) this.maskFrames.delete(this.video.id);
    this.maskKey = ""; this.maskImage = null; this.drawOverlay();
  }

  drawOverlay() {
    if (this.disposed) return;
    const canvas = this.overlay, rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width)); canvas.height = Math.max(1, Math.round(rect.height));
    const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
    const info = this.video?.info;
    const fit = frameRect({ left: 0, top: 0, width: canvas.width, height: canvas.height }, info?.width || this.player.videoWidth, info?.height || this.player.videoHeight);
    if (!fit) return;
    if (this.showMask && this.maskImage) { ctx.drawImage(this.maskImage, fit.left, fit.top, fit.width, fit.height); }
    this.loadMask();
    const shot = this.video?.shots?.find(item => item.start <= this.time && this.time < item.end);
    for (const binding of this.video?.bindings || []) {
      const selected = binding.role_id === this.role?.id;
      const anchor = binding.anchors.find(item => Math.abs(item.time - this.time) < 0.6 / this.fps &&
        (!shot || ((shot.scene_start ?? shot.start) <= item.time && item.time < (shot.scene_end ?? shot.end))));
      const inferred = anchor?.box ? null : interpolatedBox(this.video, binding, this.time);
      if (!anchor && !inferred) continue;
      ctx.strokeStyle = selected ? "#69dac7" : "#ffc96b"; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
      if (anchor?.box) this.drawBox(ctx, fit, anchor.box);
      if (inferred) { ctx.setLineDash([6, 4]); this.drawBox(ctx, fit, inferred); ctx.setLineDash([]); }
      for (const point of anchor?.points || []) {
        ctx.fillStyle = point.label === 0 ? "#ff8096" : "#69dac7"; ctx.beginPath(); ctx.arc(fit.left + point.x * fit.width, fit.top + point.y * fit.height, 4, 0, Math.PI * 2); ctx.fill();
      }
      for (const stroke of anchor?.strokes || []) this.drawStroke(ctx, fit, stroke);
      const box = anchor?.box || inferred;
      const position = box ? { x: box[0], y: box[1] } : anchor?.points?.[0] || anchor?.strokes?.[0]?.points?.[0];
      if (position) {
        const name = (this.state.roles.find(role => role.id === binding.role_id)?.name || "角色") + (inferred ? " · 同镜头插值" : "");
        ctx.font = "11px sans-serif"; ctx.fillStyle = "#07151a"; ctx.fillRect(fit.left + position.x * fit.width, fit.top + position.y * fit.height, ctx.measureText(name).width + 10, 18);
        ctx.fillStyle = selected ? "#69dac7" : "#ffc96b"; ctx.fillText(name, fit.left + position.x * fit.width + 5, fit.top + position.y * fit.height + 13);
      }
    }
    if (this.drag) {
      const d = this.drag; ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
      if (d.tool === "box") this.drawBox(ctx, fit, [d.start.x, d.start.y, d.current.x, d.current.y]);
      if (["brush", "erase"].includes(d.tool)) this.drawStroke(ctx, fit, { points: d.points, radius: 0.015, label: d.label });
    }
  }

  drawBox(ctx, fit, box) { ctx.strokeRect(fit.left + box[0] * fit.width, fit.top + box[1] * fit.height, (box[2] - box[0]) * fit.width, (box[3] - box[1]) * fit.height); }

  drawStroke(ctx, fit, stroke) {
    if (!stroke.points.length) return;
    ctx.save(); ctx.globalAlpha = 0.6; ctx.strokeStyle = stroke.label ? "#69dac7" : "#ff8096";
    ctx.lineWidth = stroke.radius * Math.min(fit.width, fit.height) * 2; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.beginPath();
    stroke.points.forEach((point, i) => { if (!i) ctx.moveTo(fit.left + point.x * fit.width, fit.top + point.y * fit.height); else ctx.lineTo(fit.left + point.x * fit.width, fit.top + point.y * fit.height); });
    if (stroke.points.length === 1) { const point = stroke.points[0]; ctx.lineTo(fit.left + point.x * fit.width + 0.1, fit.top + point.y * fit.height); }
    ctx.stroke(); ctx.restore();
  }

  loadMask() {
    if (!this.showMask || !this.video) return;
    const frames = (this.maskFrames.get(this.video.id) || []).filter(frame => !this.role || frame.role_id === this.role.id);
    const nearest = frames.reduce((best, frame) => !best || Math.abs(frame.time - this.time) < Math.abs(best.time - this.time) ? frame : best, null);
    const key = nearest?.url || "";
    if (this.maskKey === key) return;
    this.maskKey = key; this.maskImage = null;
    if (!key) return;
    const image = new Image();
    image.onload = () => {
      if (this.maskKey !== key || this.disposed) return;
      const tint = document.createElement("canvas"); tint.width = image.naturalWidth; tint.height = image.naturalHeight;
      const ctx = tint.getContext("2d"); ctx.drawImage(image, 0, 0);
      const pixels = ctx.getImageData(0, 0, tint.width, tint.height);
      for (let p = 0; p < pixels.data.length; p += 4) {
        const alpha = pixels.data[p] * 0.55; pixels.data[p] = 75; pixels.data[p + 1] = 232; pixels.data[p + 2] = 190; pixels.data[p + 3] = alpha;
      }
      ctx.putImageData(pixels, 0, 0); this.maskImage = tint; this.drawOverlay();
    };
    image.onerror = () => this.status("蒙版预览读取失败，请重新跟踪。", true);
    image.src = mediaURL(key);
  }

  renderTimeline() {
    this.timelinePanel.replaceChildren();
    const video = this.video, toolbar = element("div", "csd-timeline-toolbar");
    toolbar.append(element("span", "csd-section-title", "原片时间轴"));
    const analyse = button("按镜头自动分段", () => this.runJob("analyse", "镜头分段", result => {
      const target = this.state.videos.find(v => v.id === video.id); if (!target) return;
      applyAnalysis(target, result);
      if (this.state.settings.trial.video_id === target.id) this.state.settings.trial = { enabled: false, video_id: "", segment_id: "" };
      if (target.id === this.video?.id) this.selectedSegment = target.segments[0]?.id || "";
      this.save(true);
    })); analyse.disabled = !video;
    const split = button("在播放处分裂", () => {
      try {
        const segment = video?.segments.find(segment => this.time > segment.start && this.time < segment.end);
        this.selectedSegment = splitSegment(video, segment?.id, this.time).id;
        this.state.settings.segment_mode = "manual"; this.save(true);
      } catch (error) { this.status(error.message, true); }
    }); split.disabled = !video;
    const merge = button("合并后一段", () => {
      try {
        const removedId = video.segments[video.segments.findIndex(s => s.id === this.selectedSegment) + 1]?.id;
        mergeNext(video, this.selectedSegment);
        if (this.state.settings.trial.segment_id === removedId) this.state.settings.trial.segment_id = this.selectedSegment;
        this.state.settings.segment_mode = "manual"; this.save(true);
      } catch (error) { this.status(error.message, true); }
    }); merge.disabled = !this.segment;
    const zoom = select([["1", "适合窗口"], ["2", "2×"], ["4", "4×"], ["8", "8×"]], String(this.zoom), value => { this.zoom = Number(value); this.renderTimeline(); });
    toolbar.append(analyse, split, merge, field("缩放", zoom));
    const trial = button("试跑所选段", () => this.queueTrial(), "primary"); trial.disabled = !this.segment; toolbar.append(trial);
    if (this.state.settings.trial.enabled) toolbar.append(button("恢复完整运行", () => { this.state.settings.trial.enabled = false; this.save(true); this.status("已恢复完整视频处理；点击 ComfyUI 运行按钮执行。"); }));
    this.timelinePanel.append(toolbar);
    if (!video) { this.timelinePanel.append(element("div", "muted", "加载视频后可拖动分段边界，点击分段设置提示词和参与角色。")); return; }
    const viewport = element("div", "csd-timeline-viewport"), timeline = element("div", "csd-timeline");
    timeline.style.width = `${this.zoom * 100}%`;
    const ruler = element("div", "csd-ruler"), duration = video.info.duration || 1;
    const count = Math.min(50, Math.max(5, Math.ceil(duration / 2)));
    for (let i = 0; i <= count; i++) {
      const tick = element("div", "csd-tick"); tick.style.left = `${i / count * 100}%`;
      tick.append(element("span", "", `${(duration * i / count).toFixed(1)}s`)); ruler.append(tick);
    }
    ruler.addEventListener("pointerdown", event => {
      event.preventDefault(); const rect = timeline.getBoundingClientRect(); this.seekTo((event.clientX - rect.left) / rect.width * duration);
      ruler.setPointerCapture(event.pointerId);
    });
    ruler.addEventListener("pointermove", event => { if (ruler.hasPointerCapture(event.pointerId)) { const rect = timeline.getBoundingClientRect(); this.seekTo((event.clientX - rect.left) / rect.width * duration); } });
    ruler.addEventListener("pointerup", event => { if (ruler.hasPointerCapture(event.pointerId)) ruler.releasePointerCapture(event.pointerId); });
    timeline.append(ruler);
    video.segments.forEach((segment, index) => {
      const tile = element("div", `csd-segment${this.selectedSegment === segment.id ? " selected" : ""}${segment.enabled === false ? " disabled" : ""}`);
      tile.style.left = `${segment.start / duration * 100}%`; tile.style.width = `${(segment.end - segment.start) / duration * 100}%`;
      tile.append(element("strong", "", `片段 ${index + 1}${this.state.settings.trial.enabled && this.state.settings.trial.segment_id === segment.id ? " · 试跑" : ""}`), element("small", "", `${segment.start.toFixed(2)}—${segment.end.toFixed(2)}s`));
      tile.addEventListener("click", () => { this.selectedSegment = segment.id; this.seekTo(segment.start); this.renderTimeline(); }); timeline.append(tile);
      if (index > 0) {
        const boundary = element("div", "csd-boundary"); boundary.style.left = `${segment.start / duration * 100}%`; boundary.title = "拖动边界，两侧片段连续衔接";
        boundary.addEventListener("pointerdown", event => { event.preventDefault(); event.stopPropagation(); boundary.setPointerCapture(event.pointerId); this.boundaryDrag = index; });
        boundary.addEventListener("pointermove", event => {
          if (!boundary.hasPointerCapture(event.pointerId)) return;
          const rect = timeline.getBoundingClientRect(); moveBoundary(video, index, (event.clientX - rect.left) / rect.width * duration);
          const tiles = timeline.querySelectorAll(".csd-segment"), previous = video.segments[index - 1];
          for (const j of [index - 1, index]) {
            tiles[j].style.left = `${video.segments[j].start / duration * 100}%`; tiles[j].style.width = `${(video.segments[j].end - video.segments[j].start) / duration * 100}%`;
            tiles[j].querySelector("small").textContent = `${video.segments[j].start.toFixed(2)}—${video.segments[j].end.toFixed(2)}s`;
          }
          boundary.style.left = `${previous.end / duration * 100}%`;
        });
        boundary.addEventListener("pointerup", event => {
          boundary.releasePointerCapture(event.pointerId); this.boundaryDrag = null; this.state.settings.segment_mode = "manual"; this.save(); this.renderSettings(); this.renderTimeline();
        });
        boundary.addEventListener("pointercancel", () => { this.boundaryDrag = null; this.save(); this.renderTimeline(); }); timeline.append(boundary);
      }
    });
    this.playhead = element("div", "csd-playhead"); timeline.append(this.playhead); viewport.append(timeline); this.timelinePanel.append(viewport);
    this.timelinePanel.append(element("div", "csd-note", "拖动白色边界调整长度。首次生成时关闭的片段沿用原片；二次修正中未重跑的片段保留所选底稿。"));
    const segment = this.segment;
    if (segment) {
      const editor = element("div", "csd-segment-editor"), options = element("div", "csd-segment-options");
      options.append(check("替换此段人物", segment.enabled !== false, value => { segment.enabled = value; this.save(); this.renderTimeline(); }));
      const boundaryFields = element("div", "csd-preview-tools");
      const index = video.segments.indexOf(segment);
      if (index > 0) boundaryFields.append(field("开始", numeric(segment.start, value => { moveBoundary(video, index, value); this.save(); this.renderTimeline(); }, { max: duration })));
      if (index < video.segments.length - 1) boundaryFields.append(field("结束", numeric(segment.end, value => { moveBoundary(video, index + 1, value); this.save(); this.renderTimeline(); }, { max: duration })));
      options.append(boundaryFields);
      const roles = element("div", "csd-role-checks");
      const mode = segmentRoleMode(segment), participation = segmentParticipation(video, segment);
      options.append(field("参与角色", select([["auto", "自动按镜头"], ["manual", "手动选择"]], mode, value => {
        segment.role_mode = value;
        segment.role_ids = value === "manual" ? [...participation.role_ids] : [];
        this.save(); this.renderTimeline();
      })));
      for (const binding of video.bindings) {
        const role = this.state.roles.find(r => r.id === binding.role_id); if (!role) continue;
        const choice = check(role.name, mode === "manual" ? segment.role_ids.includes(role.id) : participation.role_ids.includes(role.id), value => {
          if (value) segment.role_ids = [...new Set([...segment.role_ids, role.id])]; else segment.role_ids = segment.role_ids.filter(id => id !== role.id);
          this.save(); this.renderTimeline();
        });
        choice.querySelector("input").disabled = mode === "auto";
        roles.append(choice);
      }
      options.append(roles);
      const roleNames = ids => ids.map(id => this.state.roles.find(role => role.id === id)?.name || "角色").join("、") || "无，沿用原片";
      const summary = element("div", "csd-note csd-role-summary");
      if (participation.known) {
        for (const row of participation.rows) summary.append(element("div", "", `${timeText(row.start)}—${timeText(row.end)}：${roleNames(row.role_ids)}`));
        summary.append(element("div", "", "按本镜头定位及下方局部人物设置选角；只露手臂或脚的角色也要纳入。"));
      } else {
        summary.textContent = `尚未分析镜头，本段已标记候选：${roleNames(participation.role_ids)}。运行时按真实镜头确认，不跨未确定镜头借用框选。`;
      }
      options.append(summary);
      const prompt = element("textarea"); prompt.value = segment.prompt || ""; prompt.placeholder = "此段补充提示词（可选）：描述外观、遮挡或动作要求";
      prompt.addEventListener("change", () => { segment.prompt = prompt.value; this.save(); }); editor.append(options, prompt); this.timelinePanel.append(editor);
      this.renderShotSettings(video, segment, editor);
    }
    this.updateTime();
  }

  renderShotSettings(video, segment, editor) {
    const shots = (video.shots || []).filter(shot => shot.start < segment.end && shot.end > segment.start);
    const panel = element("details", "csd-shot-settings");
    panel.open = this.shotSettingsOpen || false;
    panel.addEventListener("toggle", () => { this.shotSettingsOpen = panel.open; });
    panel.append(element("summary", "", "逐镜头动作与局部人物 · v0.3.3"));
    panel.append(element("div", "csd-note", "只露手、手臂、脚也属于出镜。设置“可见，参与替换”后会发送该角色参考及原片人物特征；没有定位时可补框，或写清局部位置。同一原镜头跨分段时共用这些设置。"));
    if (!shots.length) panel.append(element("div", "csd-note", "请先点击“按镜头自动分段”取得原片镜头。"));
    for (const shot of shots) {
      const range = sourceShotRange(shot), saved = shotOverride(video, shot);
      const card = element("div", "csd-shot-card");
      card.append(element("strong", "", `原片 ${timeText(range.start)}—${timeText(range.end)}`), button("定位镜头", () => this.seekTo(Math.max(segment.start, range.start))));
      for (const binding of video.bindings || []) {
        const role = this.state.roles.find(item => item.id === binding.role_id); if (!role) continue;
        const current = shotRoleOverride(video, shot, role.id);
        const row = element("div", "csd-preview-tools");
        row.append(field(`${role.name} · 本镜头`, select([["auto", "按本镜头定位"], ["visible", "可见，参与替换"], ["absent", "不参与替换"]], current?.presence || "auto", value => {
          shotRoleOverride(video, shot, role.id, true).presence = value;
          if (value === "visible" && segmentRoleMode(segment) === "manual") segment.role_ids = [...new Set([...segment.role_ids, role.id])];
          if (segmentParticipation(video, segment).rows.some(item => item.role_ids.length > 1)) this.state.settings.person_mode = "multi";
          this.save(); this.renderSettings(); this.renderTimeline();
        })));
        row.append(field(`${role.name} · 可见部位`, select([["", "按原片画面"], ["hand_arm", "只露手 / 前臂"], ["hands", "只露手"], ["feet", "只露脚"], ["torso", "躯干局部"], ["head", "头部局部"]], current?.body_part || "", value => {
          shotRoleOverride(video, shot, role.id, true).body_part = value; this.save();
        })));
        const location = element("input"); location.type = "text"; location.value = current?.description || "";
        location.placeholder = "局部位置，如左侧前景持枪手；保持脸和身体在画面外";
        location.addEventListener("change", () => { shotRoleOverride(video, shot, role.id, true).description = location.value; this.save(); });
        card.append(row, field(`${role.name} · 局部位置`, location));
      }
      const action = element("textarea"); action.value = saved?.prompt || "";
      action.placeholder = "按顺序描述本镜头：开头景别与动作 → 中间运镜、接触和遮挡 → 末尾可见部位与退场。原片时间自动映射。";
      action.addEventListener("change", () => { shotOverride(video, shot, true).prompt = action.value; this.save(); });
      card.append(field(`镜头 ${range.start.toFixed(3)} 秒动作顺序`, action));
      panel.append(card);
    }
    editor.append(panel);
  }

  renderFramePanel() {
    for (const player of this.framePanel.querySelectorAll("video")) player.pause();
    this.framePanel.replaceChildren();
    const video = this.video;
    this.framePanel.append(element("div", "csd-section-title", "画面参考 / 二次修正 · 首次生成可以留空"));
    if (!video) return;
    const actions = element("div", "csd-preview-tools");
    actions.append(field("图片用途", select([["guide", "已修正整帧 · 指定时间"], ["scene", "原片截图 · 场景动作参考"]], this.referenceKind, value => { this.referenceKind = value; })));
    const context = () => ({ kind: this.referenceKind, video_id: video.id, time: Math.min(this.time, video.info.duration - 1 / this.fps) });
    actions.append(button("上传画面参考", () => this.pickFiles("image", context())), button("加载画面参考", () => this.openAssetPicker("image", context())),
      button("H3 生成本段候选帧", () => this.queueCandidate()), button("刷新结果历史", () => this.loadHistory()));
    this.framePanel.append(actions, element("div", "csd-note", "参考时间取原片当前播放位置。原片截图不会锁住旧人物；修正整帧会约束该时间。外部修图由你完成，再上传此处。候选需生成本段视频后截帧，可选择采用。"));
    const list = element("div", "csd-frame-list");
    for (const ref of this.state.frame_references.filter(item => item.video_id === video.id)) {
      const image = this.state.images.find(item => item.id === ref.image_id);
      if (!image) continue;
      const card = element("div", "csd-frame-card"), thumbnail = element("img"); thumbnail.alt = image.name;
      try { thumbnail.src = this.assetURL(image); } catch {}
      const purpose = element("input"); purpose.value = ref.purpose || ""; purpose.placeholder = "此帧修正内容，例如背景布局";
      purpose.addEventListener("change", () => { ref.purpose = purpose.value; this.save(); });
      card.append(thumbnail, check("启用", ref.enabled !== false, value => { ref.enabled = value; this.save(); }),
        select([["scene", "场景参考"], ["guide", "修正整帧"]], ref.kind, value => { ref.kind = value; this.save(); }),
        numeric(ref.time, value => { ref.time = Math.max(0, Math.min(video.info.duration - 1 / this.fps, value)); this.save(); }, { max: video.info.duration, step: 1 / this.fps }),
        purpose, button("定位", () => this.seekTo(ref.time)), button("移除", () => { this.state.frame_references = this.state.frame_references.filter(item => item !== ref); this.save(true); }));
      list.append(card);
    }
    this.framePanel.append(list);
    const revision = this.state.settings.revision, row = element("div", "csd-preview-tools");
    row.append(check("二次修正", revision.enabled && revision.video_id === video.id, value => {
      if (revision.video_id !== video.id) revision.base_result_id = "";
      revision.enabled = value; revision.video_id = video.id; revision.segment_ids = [this.selectedSegment].filter(Boolean);
      this.state.settings.trial.enabled = false; this.state.settings.candidate.enabled = false; this.save(true);
    }));
    const full = this.history.filter(item => item.video_id === video.id && !item.report?.trial && !item.report?.candidate);
    row.append(field("保留满意片段的底稿", select([["", "选择完整生成版本"], ...full.map(item => [item.result_id, item.name])], revision.base_result_id, value => { revision.base_result_id = value; this.save(); })),
      field("重跑范围", select([["selected", "所选分段"], ["all", "整片"]], revision.scope, value => { revision.scope = value; this.save(true); })),
      button("运行当前方案", () => this.queueCurrent()));
    if (revision.enabled && revision.scope === "selected") for (const segment of video.segments) row.append(check(`${timeText(segment.start)}–${timeText(segment.end)}`, revision.segment_ids.includes(segment.id), value => {
      revision.segment_ids = value ? [...new Set([...revision.segment_ids, segment.id])] : revision.segment_ids.filter(id => id !== segment.id); this.save();
    }));
    this.framePanel.append(row);
    const history = element("details", "csd-history"); history.open = Boolean(this.history.length);
    history.append(element("summary", "", `当前视频的生成历史 · ${this.history.length}`));
    for (const result of this.history.filter(item => item.video_id === video.id)) {
      const card = element("div", "csd-result"); card.append(element("strong", "", `${result.report?.candidate ? "候选" : result.report?.trial ? "试跑" : "完整版本"} · ${result.name}`));
      const player = element("video"); player.src = mediaURL(result.preview_url || result.url); player.controls = true; player.preload = "metadata"; player.playsInline = true;
      if (result.report?.candidate) player.addEventListener("loadedmetadata", () => { player.currentTime = Math.max(0, Math.min(player.duration - 1 / this.fps, result.report.candidate_time - result.report.start)); }, { once: true });
      const absoluteTime = () => (result.report?.start || 0) + player.currentTime;
      const tools = element("div", "csd-preview-tools"); tools.append(button("截取当前帧供外部修图", () => this.exportFrame(result.result_id, absoluteTime(), false)));
      if (result.report?.candidate) tools.append(button("采用此时刻候选作为修正帧", () => this.exportFrame(result.result_id, absoluteTime(), false, true)));
      const download = element("a", "", "下载视频"); download.href = mediaURL(result.url); download.download = result.name; tools.append(download);
      card.append(player, tools); history.append(card);
      if (result.candidate_url) {
        const candidate = element("img"); candidate.src = mediaURL(result.candidate_url); candidate.alt = "所选时间的 H3 候选帧"; candidate.style.maxWidth = "100%"; card.append(candidate);
      }
    }
    this.framePanel.append(history);
  }

  async loadHistory() {
    const video = this.video;
    if (!video) return;
    try {
      const result = await request(`/csd/results?source=${encodeURIComponent(video.file)}`, { signal: this.abort.signal });
      if (this.disposed || this.video?.id !== video.id) return;
      this.history = result.results; this.renderFramePanel();
    } catch (error) { if (error.name !== "AbortError") this.status(`历史读取失败：${error.message}`, true); }
  }

  receiveResults(results) {
    const relevant = results.filter(item => item.planner_node == this.node.id);
    if (!relevant.length) return;
    this.state.settings.candidate.enabled = false;
    for (const result of relevant) {
      this.history = [result, ...this.history.filter(item => item.result_id !== result.result_id)];
      const video = this.state.videos.find(item => item.id === result.video_id && item.file === result.report?.source);
      if (video && result.report.source_shots?.length) video.shots = structuredClone(result.report.source_shots);
    }
    this.save(); this.renderTimeline(); this.drawOverlay(); this.renderFramePanel();
    const warnings = relevant.flatMap(result => result.report?.warnings || []);
    if (warnings.length) this.status([...new Set(warnings)].join("；"));
  }

  async exportFrame(resultId, time, addScene = false, adoptGuide = false) {
    const video = this.video;
    if (!video) return;
    try {
      const body = { state: structuredClone(this.state), video_id: video.id, time, result_id: resultId };
      for (const name of ["trial", "revision", "candidate"]) body.state.settings[name].enabled = false;
      const result = await request("/csd/frame", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: this.abort.signal });
      this.urls.set(result.asset.file, result.asset);
      if (addScene || adoptGuide) {
        addFrameReference(this.state, result.asset, { video_id: video.id, time: result.time, kind: adoptGuide ? "guide" : "scene" });
        this.save(true);
      }
      const dialog = this.dialog("整帧截图已保存"), image = element("img"); image.src = mediaURL(result.asset.url); image.style.maxWidth = "100%";
      const download = element("a", "", "下载 PNG 图片"); download.href = mediaURL(result.asset.url); download.download = result.asset.name;
      dialog.append(image, download, element("p", "", `原片时间 ${timeText(result.time)}。外部修图后，通过“上传画面参考”以修正整帧导入相同时间。`));
    } catch (error) { this.status(`截图失败：${error.message}`, true); }
  }

  async queueCurrent() {
    try {
      if (this.state.settings.revision.enabled && !this.state.settings.revision.base_result_id) throw new Error("请先选择上一版完整结果作为底稿。");
      this.state.settings.trial.enabled = false; this.state.settings.candidate.enabled = false; this.save(true);
      await app.queuePrompt(0, 1); this.status("当前生成方案已加入 ComfyUI 队列。");
    } catch (error) { this.status(error.message, true); }
  }

  async queueCandidate() {
    try {
      if (!this.video || !this.segment) throw new Error("请先选择分段。");
      this.state.settings.candidate = { enabled: true, video_id: this.video.id, segment_id: this.segment.id, time: Math.max(this.segment.start, Math.min(this.time, this.segment.end - 1 / this.fps)) };
      this.state.settings.revision.enabled = false; this.state.settings.trial.enabled = false; this.save(true);
      await app.queuePrompt(0, 1); this.status("H3 候选任务已排队，完成后在历史中预览并决定是否采用。");
    } catch (error) { this.state.settings.candidate.enabled = false; this.save(); this.status(error.message, true); }
  }

  async queueTrial() {
    try {
      this.state.settings.revision.enabled = false;
      selectTrial(this.state, this.video.id, this.selectedSegment); this.save(true);
      if (typeof app.queuePrompt !== "function") throw new Error("试跑范围已保存，请点击 ComfyUI 的运行按钮。");
      await app.queuePrompt(0, 1);
      this.status("所选分段已加入 ComfyUI 队列；运行后可在保存结果节点查看。再次完整运行前点“恢复完整运行”。");
    } catch (error) { this.status(error.message, true); }
  }

  pickFiles(kind, referenceContext = null) {
    const picker = element("input"); picker.type = "file"; picker.multiple = true;
    picker.accept = { video: "video/*,.mkv,.mov,.avi", image: "image/*" }[kind];
    picker.addEventListener("change", () => this.uploadFiles(picker.files, referenceContext)); picker.click();
  }

  async uploadFiles(files, referenceContext = null) {
    for (const file of Array.from(files)) {
      if (this.disposed) return;
      this.status(`正在上传 ${file.name}…`);
      try {
        const form = new FormData(); form.append("file", file);
        const asset = await request("/csd/upload", { method: "POST", body: form, signal: this.abort.signal });
        if (this.disposed) return;
        this.urls.set(asset.file, asset); const item = addAsset(this.state, asset);
        if (asset.kind === "video") this.selectedSegment = item.segments[0]?.id || "";
        if (referenceContext) addFrameReference(this.state, item, referenceContext);
        else if (asset.kind === "image" && this.role) toggleRoleImage(this.role, item.id, true);
        this.save(true); this.status(`已添加 ${asset.name}`);
      } catch (error) { if (error.name !== "AbortError") this.status(`${file.name}：${error.message}`, true); }
    }
  }

  async resolveAssets() {
    const results = await Promise.allSettled(["video", "image"].map(async kind => {
      const result = await request(`/csd/assets?kind=${kind}`, { signal: this.abort.signal }); this.assetList[kind] = result.assets;
      for (const asset of result.assets) this.urls.set(asset.file, { ...this.urls.get(asset.file), ...asset });
    }));
    if (this.disposed) return;
    for (const result of results) if (result.status === "rejected" && result.reason.name !== "AbortError") this.status(`素材列表读取失败：${result.reason.message}`, true);
    this.renderLibrary(); this.renderRoles();
    const videos = this.state.videos;
    const inspected = await Promise.allSettled(videos.map(async video => {
      const asset = await request(`/csd/asset?kind=video&file=${encodeURIComponent(video.file)}`, { signal: this.abort.signal });
      if (this.disposed || !this.state.videos.includes(video)) return;
      this.urls.set(asset.file, asset);
      video.info = asset.info;
    }));
    if (this.disposed) return;
    for (const result of inspected) if (result.status === "rejected" && result.reason.name !== "AbortError") this.status(`视频检查失败：${result.reason.message}`, true);
    this.renderLibrary(); this.updatePreview();
    this.loadHistory();
  }

  openHelp() {
    this.root.scrollTop = 0;
    const dialog = this.dialog("使用说明");
    dialog.classList.add("csd-help");
    dialog.append(element("p", "csd-help-intro", USAGE_INTRO));
    for (const topic of USAGE_TOPICS) {
      const section = element("details", "csd-help-topic"); section.open = Boolean(topic.open);
      section.append(element("summary", "", topic.title));
      const list = element(topic.ordered ? "ol" : "ul");
      for (const item of topic.items) list.append(element("li", "", item));
      section.append(list); dialog.append(section);
    }
    const close = dialog.querySelector("button"); close.focus();
    dialog.addEventListener("keydown", event => {
      if (event.key === "Escape") { event.preventDefault(); dialog.remove(); this.root.querySelector(".csd-actions button[title='查看使用说明']")?.focus(); }
    });
  }

  dialog(title) {
    this.root.querySelector(".csd-dialog")?.remove();
    const dialog = element("div", "csd-dialog"); dialog.setAttribute("role", "dialog"); dialog.setAttribute("aria-label", title);
    const heading = element("div", "csd-dialog-title"); heading.append(element("strong", "", title), button("关闭", () => dialog.remove()));
    dialog.append(heading); this.root.append(dialog); return dialog;
  }

  async openAssetPicker(kind, referenceContext = null) {
    const dialog = this.dialog(`加载${{ video: "视频", image: "图片" }[kind]}`), list = element("div", "csd-dialog-list", "正在读取输入目录…"); dialog.append(list);
    try {
      const result = await request(`/csd/assets?kind=${kind}`, { signal: this.abort.signal });
      if (!dialog.isConnected) return;
      list.replaceChildren();
      const search = element("input"); search.type = "search"; search.placeholder = "按文件名搜索"; search.style.width = "100%"; dialog.insertBefore(search, list);
      const renderList = () => {
        list.replaceChildren();
        for (const asset of result.assets.filter(asset => (asset.name + asset.file).toLowerCase().includes(search.value.toLowerCase()))) {
          this.urls.set(asset.file, { ...this.urls.get(asset.file), ...asset });
          list.append(button(asset.file, () => this.loadAsset(asset, referenceContext)));
        }
        if (!list.childElementCount) list.append(element("div", "muted", "输入目录没有匹配的素材，可使用顶部上传。"));
      };
      search.addEventListener("input", renderList); renderList();
    } catch (error) { list.textContent = error.message; }
  }

  async loadAsset(listed, referenceContext = null) {
    this.status(`正在检查 ${listed.name}…`);
    try {
      const asset = listed.kind === "video" && !listed.info ? await request(`/csd/asset?kind=video&file=${encodeURIComponent(listed.file)}`, { signal: this.abort.signal }) : listed;
      if (this.disposed) return;
      this.urls.set(asset.file, asset);
      const item = addAsset(this.state, asset);
      if (asset.kind === "video") { this.time = 0; this.selectedSegment = item.segments[0]?.id || ""; }
      if (referenceContext) addFrameReference(this.state, item, referenceContext);
      else if (asset.kind === "image" && this.role) toggleRoleImage(this.role, item.id, true);
      this.save(true); this.status(`已加载 ${asset.name}`);
    } catch (error) { if (error.name !== "AbortError") this.status(`加载失败：${error.message}`, true); }
  }

  async inspectDependencies() {
    const dialog = this.dialog("本地依赖检查"), content = element("div", "muted", "正在检查…"); dialog.append(content);
    try {
      const result = await request("/csd/dependencies", { signal: this.abort.signal });
      content.replaceChildren();
      if (result.message) content.append(element("p", "", result.message));
      for (const item of result.dependencies || []) {
        const row = element("div", "csd-role-binding"); row.append(element("span", "", `${item.available ? "✓" : "○"} ${item.name}: ${item.message || item.path || (item.available ? "可用" : "缺失")}`)); content.append(row);
      }
      if (!result.dependencies?.length) { const pre = element("pre"); pre.style.whiteSpace = "pre-wrap"; pre.textContent = JSON.stringify(result, null, 2); content.append(pre); }
    } catch (error) { content.textContent = error.message; }
  }

  async runJob(endpoint, title, apply) {
    const video = this.video;
    if (!video) { this.status("请先加载一个参考视频。", true); return; }
    if ([...this.jobs.values()].some(job => job.endpoint === endpoint && job.video_id === video.id && ["queued", "running", "cancelling"].includes(job.status))) { this.status("此视频的同类任务正在运行。"); return; }
    if (endpoint === "track" && !video.bindings.some(binding => binding.anchors.length)) { this.status("请先点选或框选至少一个视频人物。", true); return; }
    try {
      const state = structuredClone(this.state);
      for (const name of ["trial", "revision", "candidate"]) state.settings[name].enabled = false;
      const result = await request(`/csd/${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ state, video_id: video.id }), signal: this.abort.signal });
      const job = { id: result.job_id, title, endpoint, video_id: video.id, status: "queued", progress: 0, message: "排队中", apply, state };
      this.jobs.set(job.id, job); this.renderJobs(); this.pollJob(job);
    } catch (error) { if (error.name !== "AbortError") this.status(error.message, true); }
  }

  async pollJob(job) {
    if (this.disposed || ["complete", "error", "cancelled"].includes(job.status)) return;
    try {
      const result = await request(`/csd/jobs/${encodeURIComponent(job.id)}`, { signal: this.abort.signal });
      if (this.disposed) return;
      Object.assign(job, result); this.renderJobs();
      if (job.cancel_requested && ["queued", "running"].includes(job.status)) { job.status = "cancelling"; job.message = "已请求取消，等待任务停止"; this.renderJobs(); }
      if (job.status === "complete") {
        if (!this.state.videos.some(video => video.id === job.video_id)) return;
        const before = job.state.videos.find(video => video.id === job.video_id);
        const current = this.state.videos.find(video => video.id === job.video_id);
        const relevantChanged = job.endpoint === "track" ? JSON.stringify(before.bindings) !== JSON.stringify(current.bindings) : job.endpoint === "analyse" && (JSON.stringify(before.segments) !== JSON.stringify(current.segments) || JSON.stringify(before.shots) !== JSON.stringify(current.shots) || job.state.settings.max_segment_seconds !== this.state.settings.max_segment_seconds || job.state.settings.scene_threshold !== this.state.settings.scene_threshold);
        if (relevantChanged) { this.status(`${job.title}已完成，但设置在处理期间被修改，请重新执行以免覆盖新设置。`, true); return; }
        if (job.endpoint === "track") {
          this.maskFrames.set(job.video_id, result.result.mask_frames || []); this.maskKey = ""; this.showMask = true; this.drawOverlay();
          this.status(`人物跟踪完成 · ${current.name}。移动播放位置检查蒙版，必要时添加锚点修正。`);
        } else { job.apply(result.result); this.status(`${job.title}完成 · ${current.name}`); }
        return;
      }
      if (job.status === "error") { this.status(`${job.title}失败：${job.message}`, true); return; }
      if (job.status === "cancelled") { this.status(`${job.title}已取消。`); return; }
      job.timer = setTimeout(() => this.pollJob(job), 1000);
    } catch (error) {
      if (error.name !== "AbortError") { job.status = "error"; job.message = error.message; this.renderJobs(); this.status(`任务状态读取失败：${error.message}`, true); }
    }
  }

  async cancelJob(job) {
    try {
      const result = await request(`/csd/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST", signal: this.abort.signal });
      Object.assign(job, result); job.cancel_requested = true; job.message = result.message || "已请求取消，等待任务停止"; this.renderJobs();
    } catch (error) { this.status(`取消失败：${error.message}`, true); }
  }

  renderJobs() {
    this.jobPanel.replaceChildren(); this.jobPanel.hidden = !this.jobs.size;
    for (const job of this.jobs.values()) {
      const row = element("div", "csd-job"), progress = element("progress"); progress.max = 1; progress.value = Math.max(0, Math.min(1, Number(job.progress) || 0));
      row.append(progress, element("span", "", `${job.title} · ${{ queued: "排队", running: "处理中", cancelling: "取消中", complete: "完成", error: "失败", cancelled: "取消" }[job.status] || job.status} · ${job.message || ""}`));
      if (["queued", "running"].includes(job.status) && job.id !== "execution") row.append(button("取消", () => this.cancelJob(job)));
      else if (!["queued", "running", "cancelling"].includes(job.status)) row.append(button("×", () => { this.jobs.delete(job.id); this.renderJobs(); }));
      this.jobPanel.append(row);
    }
  }

  executionProgress(event) {
    const detail = event.detail || event;
    if (String(detail.node) !== String(this.node.id)) return;
    const key = "execution";
    const job = this.jobs.get(key) || { id: key, title: "ComfyUI 处理", status: "running" };
    const progress = Number(detail.progress) || 0;
    Object.assign(job, { message: detail.message || detail.stage, progress, status: detail.status || (progress >= 1 ? "complete" : "running") });
    this.jobs.set(key, job); this.renderJobs();
  }

  destroy() {
    this.disposed = true; this.abort.abort(); this.resizeObserver.disconnect(); this.player.pause();
    this.player.removeAttribute("src"); this.player.load();
    for (const job of this.jobs.values()) clearTimeout(job.timer);
    this.jobs.clear(); this.maskFrames.clear(); this.maskImage = null;
  }
}

function hidePlanWidget(node) {
  const widget = node.widgets?.find(widget => widget.name === "project_json");
  if (!widget) return;
  widget.type = "hidden"; widget.hidden = true; widget.computeSize = () => [0, -4];
  widget.options ||= {}; widget.options.hideInPanel = true;
  const input = widget.element || widget.inputEl;
  if (input) {
    input.style.setProperty("display", "none", "important"); input.style.setProperty("pointer-events", "none", "important");
    const host = input.closest(".dom-widget") || input.parentElement;
    if (host) for (const [key, value] of [["display", "none"], ["pointer-events", "none"], ["height", "0px"], ["min-height", "0px"]]) host.style.setProperty(key, value, "important");
  }
  // Nodes 2.0 can mount a Vue textarea instead of widget.element.
  for (const input of document.querySelectorAll(`[node-id="${node.id}"][node-type="CSD_MaterialPlanner"] textarea`)) {
    if (input.closest(".csd")) continue;
    if (input.placeholder === "project_json" || input.getAttribute("data-widget-name") === "project_json") {
      input.style.setProperty("display", "none", "important"); input.style.setProperty("pointer-events", "none", "important");
      input.parentElement?.style.setProperty("display", "none", "important");
    }
  }
}

function registerResultPreview(nodeType) {
  const executed = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    const previous = executed?.apply(this, arguments);
    const results = message.csd_results || message.ui?.csd_results;
    if (!Array.isArray(results)) return previous;
    for (const ui of nodes.values()) ui.receiveResults(results);
    if (!this.__csdOutput) {
      const root = element("div", "csd csd-output");
      const dom = this.addDOMWidget("csd_results_preview", "div", root, { serialize: false, hideOnZoom: false });
      dom.serialize = false;
      dom.computeSize = width => [Math.max(200, (this.size?.[0] || width || 460) - 20), Math.max(100, Math.min(900, root.scrollHeight || 330))];
      dom.options.getMinHeight = () => 100;
      dom.options.getMaxHeight = () => Math.max(100, Math.min(900, root.scrollHeight || 330));
      this.__csdOutput = root;
      requestAnimationFrame(() => root.closest(".dom-widget")?.classList.add("csd-widget-host"));
    }
    const root = this.__csdOutput;
    for (const video of root.querySelectorAll("video")) video.pause();
    root.replaceChildren();
    for (const result of results) {
      const card = element("div", "csd-result"), title = element("strong", "", result.name || "替换结果"); card.append(title);
      try {
        if (result.preview_url || result.url) {
          const video = element("video"); video.src = mediaURL(result.preview_url || result.url); video.controls = true; video.playsInline = true; video.preload = "metadata";
          video.addEventListener("error", () => { card.append(element("div", "muted", "此格式无法在浏览器播放，可下载原文件查看。")); }); card.append(video);
        }
        if (result.url) {
          const download = element("a", "", "下载结果视频"); download.href = mediaURL(result.url); download.download = result.name || ""; card.append(download);
        }
      } catch (error) { card.append(element("div", "muted", error.message)); }
      if (result.path) card.append(element("div", "csd-result-path", result.path));
      if (result.report) {
        const details = element("details"), report = element("pre", "", typeof result.report === "string" ? result.report : JSON.stringify(result.report, null, 2));
        details.append(element("summary", "", "处理记录"), report); card.append(details);
      }
      root.append(card);
    }
    requestAnimationFrame(() => { this.setSize?.([Math.max(460, this.size?.[0] || 0), this.computeSize?.()[1] || 400]); this.setDirtyCanvas?.(true, true); });
    return previous;
  };
  const removed = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function () {
    for (const video of this.__csdOutput?.querySelectorAll("video") || []) { video.pause(); video.removeAttribute("src"); video.load(); }
    this.__csdOutput = null; return removed?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "CharacterSwapDirector.MaterialPlanner",
  setup() {
    if (!document.getElementById("csd-styles")) {
      const link = element("link"); link.id = "csd-styles"; link.rel = "stylesheet"; link.href = new URL("./director.css", import.meta.url).href; document.head.append(link);
    }
    window.addEventListener("wheel", scrollDirector, { capture: true, passive: false });
    api.addEventListener("csd.progress", event => { for (const ui of nodes.values()) ui.executionProgress(event); });
  },
  onNodeOutputsUpdated(outputs) {
    for (const [id, output] of Object.entries(outputs || {})) {
      const node = app.graph?.getNodeById(id);
      if (node?.type === "CSD_SaveResults" && Array.isArray(output.csd_results)) node.onExecuted?.(output);
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "CSD_SaveResults") { registerResultPreview(nodeType); return; }
    if (nodeData.name !== "CSD_MaterialPlanner") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const widget = this.widgets?.find(widget => widget.name === "project_json"); hidePlanWidget(this);
      const root = element("div"), dom = this.addDOMWidget("director", "div", root, { serialize: false, hideOnZoom: false, getMinHeight: () => UI_HEIGHT, getMaxHeight: () => UI_HEIGHT });
      dom.serialize = false;
      dom.computeSize = width => [Math.max(100, (this.size?.[0] || width || 980) - 20), UI_HEIGHT];
      this.__csd = new CharacterSwapDirector(this, root, widget); nodes.set(this, this.__csd);
      for (const delay of [0, 100, 600]) setTimeout(() => {
        if (!this.__csd || this.__csd.disposed) return;
        hidePlanWidget(this); root.closest(".dom-widget")?.classList.add("csd-widget-host");
      }, delay);
      requestAnimationFrame(() => { this.setSize?.([Math.max(900, this.size?.[0] || 0), Math.max(UI_HEIGHT + 70, this.computeSize?.()[1] || 0)]); this.setDirtyCanvas?.(true, true); });
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configured?.apply(this, arguments);
      const widget = this.widgets?.find(widget => widget.name === "project_json");
      const value = info?.widgets_values_named?.project_json ?? info?.widgets_values?.[0];
      if (widget && value !== undefined) widget.value = value;
      hidePlanWidget(this); this.__csd?.reload();
      setTimeout(() => this.__csd?.resolveAssets(), 0); return result;
    };
    const resized = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () { const result = resized?.apply(this, arguments); hidePlanWidget(this); this.__csd?.drawOverlay(); return result; };
    const serialized = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      if (this.__csd && this.widgets) {
        const index = this.widgets.findIndex(widget => widget.name === "project_json");
        if (index >= 0) {
          const value = JSON.stringify(this.__csd.state); this.widgets[index].value = value;
          if (info.widgets_values) info.widgets_values[index] = value;
        }
      }
      return serialized?.apply(this, arguments);
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.__csd?.destroy(); nodes.delete(this); return removed?.apply(this, arguments); };
  },
});
