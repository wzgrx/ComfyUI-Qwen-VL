from __future__ import annotations
import torch
import os
import tempfile
import io
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_vl_utils import process_vision_info
import numpy as np
import requests
import time
import torchvision


# 模型注册表 - 存储所有支持的模型版本信息
MODEL_REGISTRY = {
    "Qwen2.5-VL-3B-Instruct": {
        "repo_id": {
            "huggingface": "Qwen/Qwen2.5-VL-3B-Instruct",
            "modelscope": "qwen/Qwen2.5-VL-3B-Instruct"
        },
        "required_files": [
            "chat_template.json", "merges.txt","model.safetensors.index.json",
            "preprocessor_config.json", "tokenizer.json", "vocab.json",
            "config.json","generation_config.json","tokenizer_config.json",
            # 3B模型分片为2个
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "test_file": "model-00002-of-00002.safetensors",
        "default": True
    },
    "Qwen2.5-VL-3B-Instruct-AWQ": {
        "repo_id": {
            "huggingface": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            "modelscope": "qwen/Qwen2.5-VL-3B-Instruct-AWQ"
        },
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "preprocessor_config.json", "tokenizer_config.json",
            "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            # 3B模型有1个分片
            "model.safetensors",
        ],
        "test_file": "model.safetensors",
        "default": False
    }
}


def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU
    except ImportError:
        return False


FLASH_ATTENTION_AVAILABLE = check_flash_attention()


def init_qwen_paths(model_name):
    """初始化模型路径，支持动态生成不同模型版本的路径"""
    base_dir = Path(folder_paths.models_dir).resolve()
    qwen_dir = base_dir / "Qwen" / "VLM"  # 添加VLM子目录
    model_dir = qwen_dir / model_name  # 使用模型名称作为子目录
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册到ComfyUI
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(model_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(model_dir)], {'.safetensors', '.bin'})
    
    print(f"模型路径已初始化: {model_dir}")
    return str(model_dir)


def test_download_speed(url):
    """测试下载速度，下载 5 秒"""
    try:
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=10)
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            if time.time() - start_time > 5:
                break
            downloaded_size += len(data)
        end_time = time.time()
        speed = downloaded_size / (end_time - start_time) / 1024  # KB/s
        return speed
    except Exception as e:
        print(f"测试下载速度时出现错误: {e}")
        return 0


def validate_model_path(model_path, model_name):
    """验证模型路径的有效性和模型文件是否齐全"""
    path_obj = Path(model_path)
    
    # 基本路径检查
    if not path_obj.is_absolute():
        print(f"错误: {model_path} 不是绝对路径")
        return False
    
    if not path_obj.exists():
        print(f"模型目录不存在: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"错误: {model_path} 不是目录")
        return False
    
    # 检查模型文件是否齐全
    if not check_model_files_exist(model_path, model_name):
        print(f"模型文件不完整: {model_path}")
        return False
    
    return True


def check_model_files_exist(model_dir, model_name):
    """检查特定模型版本所需的文件是否齐全"""
    if model_name not in MODEL_REGISTRY:
        print(f"错误: 未知模型版本 {model_name}")
        return False
    
    required_files = MODEL_REGISTRY[model_name]["required_files"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True


# 视频处理工具类
class VideoProcessor:
    def __init__(self):
        # 尝试导入torchcodec作为备选视频处理库
        self.use_torchcodec = False
        try:
            import torchcodec
            self.use_torchcodec = True
            print("使用torchcodec进行视频处理")
        except ImportError:
            print("torchcodec不可用，使用torchvision进行视频处理（有弃用警告）")
            # 抑制torchvision视频API弃用警告
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
    
    def read_video(self, video_path):
        """读取视频文件并返回帧数据"""
        start_time = time.time()
        try:
            if self.use_torchcodec:
                # 使用torchcodec读取视频
                import torchcodec
                decoder = torchcodec.VideoDecoder(video_path)
                frames = []
                for frame in decoder:
                    frames.append(frame)
                fps = decoder.get_fps()
                total_frames = len(frames)
                frames = torch.stack(frames) if frames else torch.zeros(0)
            else:
                # 使用torchvision读取视频（弃用API）
                frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
                fps = info["video_fps"]
                total_frames = frames.shape[0]
            
            process_time = time.time() - start_time
            print(f"视频处理完成: {video_path}, 总帧数: {total_frames}, FPS: {fps:.2f}, 处理时间: {process_time:.3f}s")
            return frames, fps, total_frames
            
        except Exception as e:
            print(f"视频处理错误: {e}")
            return None, None, None


class QwenVisionParser:
    def __init__(self):
        # 默认使用注册表中的第一个默认模型
        default_model = next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                            list(MODEL_REGISTRY.keys())[0])
        
        # 重置环境变量，避免干扰
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)     

        self.current_model_name = default_model
        self.current_quantization = None  # 记录当前的量化配置
        self.model_path = init_qwen_paths(self.current_model_name)
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"模型路径: {self.model_path}")
        print(f"缓存路径: {self.cache_dir}")
        
        # 验证并创建缓存目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.video_processor = VideoProcessor()  # 初始化视频处理器
        self.last_generated_text = ""  # 保存上次生成的文本，用于调试
        self.generation_stats = {"count": 0, "total_time": 0}  # 统计生成性能

    def clear_model_resources(self):
        """释放当前模型占用的资源"""
        if self.model is not None:
            print("释放当前模型占用的资源...")
            del self.model, self.processor, self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()  # 清理GPU缓存
    
    def load_model(self, model_name, quantization):
        # 检查是否需要重新加载模型
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization):
            print(f"使用已加载的模型: {model_name}，量化: {quantization}")
            return
        
        # 需要重新加载，先释放现有资源
        self.clear_model_resources()
        
        # 更新当前模型名称和路径
        self.current_model_name = model_name
        self.model_path = init_qwen_paths(self.current_model_name)
        self.current_quantization = quantization
        
        # 添加CUDA可用性检查
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for  {model_name} model")

        quant_config = None
        # 根据模型类型选择量化配置方式
        if "-AWQ" in model_name:
            # 处理 AWQ 模型（直接传递量化参数）
            quant_args = {}
            if quantization == "👍 4-bit (VRAM-friendly)":
                quant_args.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            elif quantization == "⚖️ 8-bit (Balanced Precision)":
                quant_args.update({
                    "load_in_8bit": True,
                })
        else:
            # 处理非 AWQ 模型（保持原有 BitsAndBytesConfig 逻辑）
            quant_config = None
            if quantization == "👍 4-bit (VRAM-friendly)":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif quantization == "⚖️ 8-bit (Balanced Precision)":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            quant_args = {"quantization_config": quant_config}

        # 自定义device_map，这里假设只有一个GPU，将模型尽可能放到GPU上
        device_map = {"": 0} if torch.cuda.device_count() > 0 else "auto"

        # 检查模型文件是否存在且完整
        if not validate_model_path(self.model_path, self.current_model_name):
            print(f"检测到模型文件缺失，正在为你下载 {model_name} 模型，请稍候...")
            print(f"下载将保存在: {self.model_path}")
            
            # 开始下载逻辑
            try:
                # 从注册表获取模型信息
                model_info = MODEL_REGISTRY[model_name]
                
                # 测试下载速度
                huggingface_test_url = f"https://huggingface.co/{model_info['repo_id']['huggingface']}/resolve/main/{model_info['test_file']}"
                modelscope_test_url = f"https://modelscope.cn/api/v1/models/{model_info['repo_id']['modelscope']}/repo?Revision=master&FilePath={model_info['test_file']}"
                huggingface_speed = test_download_speed(huggingface_test_url)
                modelscope_speed = test_download_speed(modelscope_test_url)

                print(f"Hugging Face下载速度: {huggingface_speed:.2f} KB/s")
                print(f"ModelScope下载速度: {modelscope_speed:.2f} KB/s")

                # 根据下载速度选择优先下载源
                if huggingface_speed > modelscope_speed * 1.5:
                    download_sources = [
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face"),
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope")
                    ]
                    print("基于下载速度分析，优先尝试从Hugging Face下载")
                else:
                    download_sources = [
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope"),
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face")
                    ]
                    print("基于下载速度分析，优先尝试从ModelScope下载")

                max_retries = 3
                success = False
                final_error = None
                used_cache_path = None

                for download_func, repo_id, source in download_sources:
                    for retry in range(max_retries):
                        try:
                            print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）...")
                            if download_func == snapshot_download:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    ignore_patterns=["*.msgpack", "*.h5"],
                                    resume_download=True,
                                    local_files_only=False
                                )
                            else:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    revision="master"
                                )

                            used_cache_path = cached_path  # 记录使用的缓存路径
                            
                            # 将下载的模型复制到模型目录
                            self.copy_cached_model_to_local(cached_path, self.model_path)
                            
                            print(f"成功从 {source} 下载模型到 {self.model_path}")
                            success = True
                            break

                        except Exception as e:
                            final_error = e  # 保存最后一个错误
                            if retry < max_retries - 1:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试...")
                            else:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源...")
                    if success:
                        break
                else:
                    raise RuntimeError("从所有源下载模型均失败。")
                
                # 下载完成后再次验证
                if not validate_model_path(self.model_path, self.current_model_name):
                    raise RuntimeError(f"下载后模型文件仍不完整: {self.model_path}")
                
                print(f"模型 {model_name} 已准备就绪")
                
            except Exception as e:
                print(f"下载模型时发生错误: {e}")
                
                # 下载失败提示
                if used_cache_path:
                    print("\n⚠️ 注意：下载过程中创建了缓存文件")
                    print(f"缓存路径: {used_cache_path}")
                    print("你可以前往此路径删除缓存文件以释放硬盘空间")
                
                raise RuntimeError(f"无法下载模型 {model_name}，请手动下载并放置到 {self.model_path}")

        # 模型文件完整，正常加载
        print(f"加载模型: {self.model_path}，量化: {quantization}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            offload_state_dict=True,
            **quant_args,  # 统一传递量化参数
        ).eval()

        # 编译优化（PyTorch 2.2+）
        if torch.__version__ >= "2.2":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # SDP优化
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 修复rope_scaling配置警告
        if hasattr(self.model.config, "rope_scaling"):
            self.model.config.rope_scaling["mrope_section"] = "none"  # 禁用 MROPE 优化

    def copy_cached_model_to_local(self, cached_path, target_path):
        """将缓存的模型文件复制到目标路径"""
        print(f"正在将模型从缓存复制到: {target_path}")
        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 使用shutil进行递归复制
        import shutil
        for item in Path(cached_path).iterdir():
            if item.is_dir():
                shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path / item.name)
        
        # 验证复制是否成功
        if validate_model_path(target_path, self.current_model_name):
            print(f"模型已成功复制到 {target_path}")
        else:
            raise RuntimeError(f"复制后模型文件仍不完整: {target_path}")

    def tensor_to_pil(self, image_tensor):
        """将图像张量转换为PIL图像"""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def preprocess_image(self, image):
        """预处理图像，包括尺寸调整和优化"""
        pil_image = self.tensor_to_pil(image)
        
        # 限制最大尺寸，避免过大的输入
        max_res = 1024
        if max(pil_image.size) > max_res:
            pil_image.thumbnail((max_res, max_res))
        
        # 转换回张量并归一化
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 转回PIL图像
        pil_image = Image.fromarray((img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return pil_image

    def preprocess_video(self, video_path):
        """预处理视频，包括帧提取和尺寸调整"""
        # 使用视频处理器读取视频
        frames, fps, total_frames = self.video_processor.read_video(video_path)
        
        if frames is None:
            print(f"无法处理视频: {video_path}")
            return None, None, None
        
        # 更激进的帧数量限制
        max_frames = 15  # 从50减少到30
        if total_frames > max_frames:
            # 采样帧
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames = frames[indices]
            print(f"视频帧数量从 {total_frames} 采样到 {len(frames)}")
        
        # 更小的帧尺寸
        resized_frames = []
        for frame in frames:
            # 转换为PIL图像
            frame_pil = Image.fromarray(frame.numpy())
            # 调整大小为384x384 (原为512x512)
            frame_pil.thumbnail((384, 384))
            # 转回张量
            frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1)
            resized_frames.append(frame_tensor)
        
        # 转换回张量
        if resized_frames:
            resized_frames = torch.stack(resized_frames)
        else:
            resized_frames = torch.zeros(0)
        
        return resized_frames, fps, len(frames)  # 返回实际采样后的帧数

    @torch.no_grad()
    def process(self, model_name, quantization, prompt, max_tokens, temperature, top_p,
                repetition_penalty, image=None, video_path=None):
        start_time = time.time()
        
        # 确保加载正确的模型和量化配置
        self.load_model(model_name, quantization)
        
        # 图像预处理
        pil_image = None
        if image is not None:
            pil_image = self.preprocess_image(image)
        
        # 视频预处理
        video_frames = None
        if video_path:
            video_frames, video_fps, video_frames_count = self.preprocess_video(video_path)
            if video_frames is not None:
                print(f"视频已处理: {video_path}, 帧数: {video_frames_count}, FPS: {video_fps}")
        
        # 构建对话
        SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving visual inputs and generating text."
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": []}
        ]
        
        # 添加图像和视频到对话
        if pil_image is not None:
            conversation[-1]["content"].append({"type": "image", "image": pil_image})
        
        if video_path and video_frames is not None:
            # 转换视频帧为PIL图像列表
            video_frame_list = []
            for frame in video_frames:
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype(np.uint8)
                video_frame_list.append(Image.fromarray(frame))
            
            conversation[-1]["content"].append({"type": "video", "video": video_frame_list})
        
        # 处理用户提示
        user_prompt = prompt if prompt.endswith(("?", ".", "！", "。", "？", "！")) else f"{prompt} "
        conversation[-1]["content"].append({"type": "text", "text": user_prompt})
        
        # 应用聊天模板
        input_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # 准备处理器参数
        processor_args = {
            "text": input_text,
            "return_tensors": "pt",
            "padding": True,
        }
        
        # 调用多模态处理逻辑（关键修正：解包两个值）
        images, videos = process_vision_info(conversation)  # 改为解包两个值
        processor_args["images"] = images
        processor_args["videos"] = videos
        
        # 清理不再需要的大对象
        del video_frames, images, videos
        torch.cuda.empty_cache()
        
        # 后续代码保持不变...
        
        # 在函数开始处初始化model_inputs为None
        model_inputs = None
        
        # 将输入移至设备
        try:
            inputs = self.processor(**processor_args).to(self.model.device)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if v is not None
            }
            
            # 确保model_inputs包含所需的键
            if "input_ids" not in model_inputs:
                raise ValueError("处理后的输入不包含'input_ids'键")
            
        except Exception as e:
            print(f"处理输入时发生错误: {e}")
            # 这里可以添加更多的错误处理逻辑，例如返回默认值或抛出特定异常
            raise RuntimeError("无法处理模型输入") from e
        
        # 生成配置
        generate_config = {
            "max_new_tokens": max(max_tokens, 10),
            "temperature": temperature,
            "do_sample": True,
            "use_cache": True,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            pre_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成前GPU内存使用: {pre_forward_memory:.2f} MB")
        
        # 检查model_inputs是否已正确初始化
        if model_inputs is None:
            raise RuntimeError("模型输入未正确初始化")

        # 使用新的autocast API
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model.generate(**model_inputs, **generate_config)

        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            post_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成后GPU内存使用: {post_forward_memory:.2f} MB")
            print(f"生成过程中GPU内存增加: {post_forward_memory - pre_forward_memory:.2f} MB")
        
        # 处理输出
        text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
        
        # 清理不再需要的大对象
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # 截取新生成的token
        input_length = model_inputs["input_ids"].shape[1]
        text_tokens = text_tokens[:, input_length:]  # 截取新生成的token
        
        # 解码文本
        text = self.tokenizer.decode(
            text_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 保存生成的文本用于调试
        self.last_generated_text = text
        del model_inputs
        torch.cuda.empty_cache()
        
        # 计算处理时间
        process_time = time.time() - start_time
        self.generation_stats["count"] += 1
        self.generation_stats["total_time"] += process_time
        
        # 打印性能统计
        print(f"生成完成，耗时: {process_time:.2f} 秒")
        if self.generation_stats["count"] > 0:
            avg_time = self.generation_stats["total_time"] / self.generation_stats["count"]
            print(f"平均生成时间: {avg_time:.2f} 秒/次")
        
        return (text.strip(),)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(MODEL_REGISTRY.keys()),  # 动态生成模型选项
                    {
                        "default": next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                                       list(MODEL_REGISTRY.keys())[0]),
                        "tooltip": "Select the available model version."
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage, suitable for resource-constrained environments.\n⚖️ 8-bit: Strikes a balance between precision and performance.\n🚫 None: Uses the original floating-point precision (requires a high-end GPU)."
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "multiline": True,
                        "tooltip": "Enter a text prompt, supporting Chinese and emojis. Example: 'Describe a cat in a painter's style.'"
                    }
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 132,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "display": "slider",
                        "tooltip": "Control the maximum length of the generated text (in tokens). \nGenerally, 100 tokens correspond to approximately 50 - 100 Chinese characters or 67 - 100 English words, but the actual number may vary depending on the text content and the model's tokenization strategy. \nRecommended range: 64 - 512."
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Control the generation diversity:\n▫️ 0.1 - 0.3: Generate structured/technical content.\n▫️ 0.5 - 0.7: Balance creativity and logic.\n▫️ 0.8 - 1.0: High degree of freedom (may produce incoherent content)."
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Nucleus sampling threshold:\n▪️ Close to 1.0: Retain more candidate words (more random).\n▪️ 0.5 - 0.8: Balance quality and diversity.\n▪️ Below 0.3: Generate more conservative content."
                    }
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Control of repeated content:\n⚠️ 1.0: Default behavior.\n⚠️ >1.0 (Recommended 1.2): Suppress repeated phrases.\n⚠️ <1.0 (Recommended 0.8): Encourage repeated emphasis."
                    }
                )
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Upload a reference image (supports PNG/JPG), and the model will adjust the generation result based on the image content."
                    }
                ),
                "video_path": (
                    "VIDEO_PATH",
                    {
                        "tooltip": "Enter the video file  (supports MP4/WEBM), and the model will extract visual features to assist in generation."
                    }
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🐼QwenVL"    




# Register the node
NODE_CLASS_MAPPINGS = {
    "QwenVisionParser": QwenVisionParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVisionParser": "Qwen-TextGraph-VisionParser🐼"
}