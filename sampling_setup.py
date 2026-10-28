"""Read the connected workflow's sampling choices without changing them."""


def sampling_setup(prompt, node_id, steps, reference_image_size):
    info = {"steps": steps, "reference_image_size": reference_image_size}
    warnings = []
    if not prompt:
        return info, warnings
    inputs = prompt.get(str(node_id), {}).get("inputs", {})
    for socket, setting in (("sampler", "sampler_name"), ("sigmas", "scheduler")):
        link = inputs.get(socket)
        if isinstance(link, list) and len(link) == 2:
            value = prompt.get(str(link[0]), {}).get("inputs", {}).get(setting)
            if value is not None:
                info[setting] = value
    loras, seen = [], set()
    link = inputs.get("model")
    while isinstance(link, list) and len(link) == 2 and str(link[0]) not in seen:
        seen.add(str(link[0]))
        node = prompt.get(str(link[0]), {})
        options = node.get("inputs", {})
        if node.get("class_type") in {"LoraLoaderModelOnly", "LoraLoader"} and options.get("strength_model", 0) != 0:
            loras.append({"name": options["lora_name"], "strength": options["strength_model"]})
        if node.get("class_type") == "UNETLoader":
            info["model_file"] = options.get("unet_name")
        link = options.get("model")
    info["loras"] = loras
    if steps < 20 and not loras:
        warnings.append(f"当前仅 {steps} 步，连接的基础 H3 模型未启用加速 LoRA；建议基础模型 20 步，或使用配套 Ref2V Turbo 工作流。插件保留外部采样设置。")
    if any(isinstance(item["name"], str) and "ref2v_turbo_8step" in item["name"].lower() for item in loras):
        if isinstance(info.get("sampler_name"), str) and info["sampler_name"] != "euler":
            warnings.append("Ref2V Turbo 8 步 LoRA 建议配套 euler 采样器；当前外部采样器未改动。")
        if reference_image_size != "match":
            warnings.append("Ref2V Turbo 参考图建议选择 match，按生成画幅缩放；当前参考图设置未改动。")
    return info, warnings
