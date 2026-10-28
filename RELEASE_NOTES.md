## H3 人物替换导演台 v0.3.3

在一个导演台内管理原视频、角色图片、人物绑定与分段计划，使用 MiniMax H3 Ref2VA 进行全画面参考生成。

### 本版内容

- 单角色和多角色的原片人物匹配描述。
- 逐镜头可见角色、局部身体部位与位置设置。
- 逐镜头动作顺序、构图、运镜和遮挡说明。
- 多张角色参考、指定时间修正整帧和参考名额管理。
- 单段试跑、原片时间映射、选择底稿和分段二次修正。
- 宽高与百万像素联动、原声/静音、MOV/MP4 和持久预览。

### 安装

将 ZIP 内的 `ComfyUI-CharacterSwapDirector` 文件夹放入 `ComfyUI/custom_nodes/`。使用运行 ComfyUI 的 Python 执行：

```bash
python ComfyUI/custom_nodes/ComfyUI-CharacterSwapDirector/setup_dependencies.py --install --verify
```

重启 ComfyUI、刷新浏览器，再拖入本页附带的 JSON 工作流并选择本机模型。升级前先保存当前工作流；安装时替换原插件目录，避免同时加载两个副本。

### 工作流

- `character_swap_local.json`：基础 Ref2VA，20 步。
- `character_swap_turbo8_local.json`：配套 Ref2V Turbo LoRA，8 步。

模型权重、私人素材和生成结果不包含在安装包中。生成效果受原素材、参考图、提示词及采样配置影响，建议先运行单段试跑。

中文/英文首页、模型目录表及完整使用说明见仓库。
