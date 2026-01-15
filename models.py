import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    if not past_key_values:
        return 0
    
    # Handle Cache objects (DynamicCache, etc.)
    if Cache is not None and isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length()
    
    # Handle legacy tuple format
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            raise NotImplementedError("vLLM backend with ModelWrapper is not fully implemented yet.")
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                # Try to use flash_attention_2 if available, fallback to auto
                try:
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                        device_map={"": device2},
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2",
                    ).eval()
                    print(f"[ModelWrapper] Loaded HF model with flash_attention_2")
                except Exception as e:
                    print(f"[ModelWrapper] Could not load with flash_attention_2: {e}")
                    print(f"[ModelWrapper] Falling back to default attention implementation")
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    ).eval()
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            # Try to use flash_attention_2 if available, fallback to auto
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )
                print(f"[ModelWrapper] Loaded model with flash_attention_2")
            except Exception as e:
                print(f"[ModelWrapper] Could not load with flash_attention_2: {e}")
                print(f"[ModelWrapper] Falling back to default attention implementation")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

        # Diagnostic: Check actual device placement
        print(f"[ModelWrapper] Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'N/A'}")
        print(f"[ModelWrapper] First parameter device: {next(self.model.parameters()).device}")
        print(f"[ModelWrapper] lm_head device: {self.model.lm_head.weight.device if hasattr(self.model, 'lm_head') else 'N/A'}")
        print(f"[ModelWrapper] Embedding device: {self.model.get_input_embeddings().weight.device}")
        
    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()
        
        # Build identity matrix for non-realign case
        identity_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, identity_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            realign_matrix, identity_matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            realign_matrix, identity_matrix, target_norm = info
            if realign_matrix.device != target_device:
                realign_matrix = realign_matrix.to(target_device)
                identity_matrix = identity_matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=realign_matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=realign_matrix.dtype)
        self._latent_realign_matrices[key] = (realign_matrix, identity_matrix, target_norm)

        return realign_matrix, identity_matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module, latent_space_realign: bool = False) -> torch.Tensor:
        realign_matrix, identity_matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        # Choose the appropriate matrix based on the per-request flag
        matrix = realign_matrix if latent_space_realign else identity_matrix
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Note: self.pre_aligned is only set when needed for ablation studies
        # to avoid memory accumulation during normal operation
        # pre_aligned = aligned.detach().clone()
        # self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Original generate_latent_batch for backward compatibility."""
        past, _ = self.generate_latent_batch_with_tokens(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            max_latent_steps=latent_steps,
        )
        return past

    @torch.no_grad()
    def generate_latent_batch_with_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: Optional[int] = None,
        past_key_values: Optional[Tuple] = None,
        max_latent_steps: int = 256,
        latent_space_realign: bool = False,
    ) -> Tuple[Tuple, List[List[int]]]:
        """
        Generate latent representations while also decoding tokens at each step.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask
            latent_steps: Fixed number of latent steps. If None, continue until EOS.
            past_key_values: Optional existing KV cache
            max_latent_steps: Maximum steps when latent_steps is None
            latent_space_realign: Whether to apply latent space realignment
            
        Returns:
            Tuple of:
            - past_key_values: Updated KV cache
            - generated_token_ids: List of token ID lists for each batch item
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        batch_size = input_ids.shape[0]
        device = self.device
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        # Get lm_head for token decoding
        lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else self.model.get_output_embeddings()
        eos_token_id = self.tokenizer.eos_token_id
        
        # Track generated tokens and completion status per batch item
        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        # Determine number of steps
        use_dynamic_stop = (latent_steps is None)
        num_steps = max_latent_steps if use_dynamic_stop else latent_steps

        for step in range(num_steps):
            # Decode token from current hidden state (just a matrix multiply)
            logits = lm_head(last_hidden)  # [B, vocab_size]
            token_ids = logits.argmax(dim=-1)  # [B]
            
            # Record tokens and check EOS
            all_finished = True
            for b in range(batch_size):
                if not finished[b]:
                    tid = token_ids[b].item()
                    generated_tokens[b].append(tid)
                    if use_dynamic_stop and tid == eos_token_id:
                        finished[b] = True
                if not finished[b]:
                    all_finished = False
            
            # Early exit if all sequences finished
            if use_dynamic_stop and all_finished:
                break

            # Apply latent realignment and continue
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model, latent_space_realign)
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past, generated_tokens
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model, self.latent_space_realign)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

    # ==================== API-friendly methods ====================
    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)
    
    @torch.no_grad()
    def generate_latent_for_api(
        self,
        prompt: str,
        *,
        latent_steps: Optional[int] = None,
        past_key_values: Optional[Tuple] = None,
        add_think_token: bool = False,
        max_latent_steps: int = 256,
        debug_max_tokens: Optional[int] = None,
        debug_continuation_prompt: Optional[str] = None,
        latent_space_realign: bool = False,
        latent_only: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[Tuple, str, int, List[int]]:
        """
        Generate latent representations for API usage.
        
        Args:
            prompt: The text prompt (already formatted with chat template)
            latent_steps: Number of latent reasoning steps. If None, continue until EOS.
            past_key_values: Optional existing KV cache to extend
            add_think_token: Whether to append <think> token
            max_latent_steps: Maximum steps when latent_steps is None
            debug_max_tokens: Max tokens to generate for debug output
            debug_continuation_prompt: Prompt to use for debug text generation. If None, uses empty/space.
            latent_space_realign: Whether to apply latent space realignment
            
        Returns:
            Tuple of:
            - Updated past_key_values (KV cache)
            - Generated text (actual model continuation using KV cache, for debugging)
            - Number of latent steps actually taken
            - Raw token IDs from latent steps (for debugging)
        """
        if add_think_token:
            prompt = f"{prompt}<think>"
        
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        prev_past_len = _past_length(past_key_values)
        
        new_past_kv, generated_token_ids = self.generate_latent_batch_with_tokens(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            max_latent_steps=max_latent_steps,
            latent_space_realign=latent_space_realign,
        )
        
        # Get actual number of steps (tokens generated)
        actual_steps = len(generated_token_ids[0])
        
        if latent_only:
            new_past_len = _past_length(new_past_kv)
            tokens_added = new_past_len - prev_past_len
            tokens_to_keep = latent_steps if latent_only else tokens_added
            output_past_kv = self._truncate_past(new_past_kv, tokens_to_keep)
        else:
            output_past_kv = new_past_kv
        if debug_max_tokens and debug_max_tokens > 0:
            # Generate debug text (don't save the returned KV cache)
            debug_texts, _ = self.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=debug_max_tokens,
                temperature=temperature,
                top_p=top_p,
                past_key_values=past_key_values,  # Use the latent-accumulated KV cache
            )
            debug_text = debug_texts[0].strip() if debug_texts else ""
        else:
            debug_text = None
        
        return output_past_kv, debug_text, actual_steps, generated_token_ids[0]
    
    @torch.no_grad()
    def generate_text_for_api(
        self,
        prompt: str,
        *,
        past_key_values: Optional[Tuple] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        add_think_token: bool = False,
    ) -> str:
        """
        Generate text for API usage, optionally using cached KV values.
        
        Args:
            prompt: The text prompt (already formatted with chat template)
            past_key_values: Optional KV cache from previous latent generations
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            add_think_token: Whether to append <think> token
            
        Returns:
            Generated text string
        """
        if add_think_token:
            prompt = f"{prompt}<think>"
        
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        generated_batch, _ = self.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            past_key_values=past_key_values,
        )
        
        return generated_batch[0].strip()

    # ==================== Hybrid vLLM + HF API methods ====================
    
    @torch.no_grad()
    def generate_latent_with_embeddings_for_api(
        self,
        prompt: str,
        *,
        latent_steps: int = 10,
        add_think_token: bool = False,
        latent_space_realign: bool = False,
        past_embeddings: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple] = None,
        embedding_insert_marker: str = "<|im_start|>user\n",
        latent_only: bool = False,
    ) -> Tuple[torch.Tensor, int, List[int], Optional[Tuple]]:
        """
        Generate latent representations and return embeddings for hybrid vLLM mode.
        
        This method uses HuggingFace to generate latent hidden states, which can
        later be injected into vLLM for text generation.
        
        OPTIMIZATION: When past_kv is provided, we skip reprocessing past embeddings
        and only process the new prompt tokens. This gives O(new_tokens) complexity
        instead of O(total_history).
        
        Args:
            prompt: The text prompt (already formatted with chat template)
            latent_steps: Number of latent reasoning steps
            add_think_token: Whether to append <think> token
            latent_space_realign: Whether to apply latent space realignment
            past_embeddings: Tensor of shape [1, L, H] from previous latent calls.
                            Used when past_kv is not available.
            past_kv: HuggingFace KV cache tuple from previous calls.
                    When provided, past_embeddings are NOT reprocessed (only stored).
            embedding_insert_marker: String marker after which to insert past embeddings
            latent_only: If True, only return the latent step embeddings (exclude prompt embeddings)
            
        Returns:
            Tuple of:
            - latent_embeddings: Tensor of shape [1, L, H] containing embeddings.
                                If latent_only=False: combined input + new latent steps
                                If latent_only=True: only new latent step embeddings
            - actual_steps: Number of NEW latent steps taken in this call
            - generated_token_ids: Raw token IDs decoded from latent steps (for debugging)
            - past_kv: Updated HuggingFace KV cache for next call
        """
        if add_think_token:
            prompt = f"{prompt}<think>"
        
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        
        # Get embedding layer and compute input embeddings
        embedding_layer = self.model.get_input_embeddings()
        input_embeddings = embedding_layer(input_ids)  # [1, L, H]
        
        # OPTIMIZATION: If we have past_kv, use it directly instead of reprocessing past_embeddings
        if past_kv is not None:
            # We have KV cache - just process new prompt tokens
            past_len = _past_length(past_kv)
            
            # Attention mask: past KV tokens + new input tokens
            attention_mask = torch.ones(
                (1, past_len + input_embeddings.shape[1]),
                dtype=torch.long,
                device=self.device,
            )
            
            # Forward pass with only new input, using cached KV
            outputs = self.model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # For embedding record: past_embeddings already stored, just add new input
            if not latent_only:
                embedding_record = [past_embeddings.to(self.device), input_embeddings.detach()]
            else:
                embedding_record = []
        
        elif past_embeddings is not None:
            # No KV cache but have past embeddings - must reprocess (slower path)
            past_embeddings = past_embeddings.to(self.device)
            
            # Find insertion point for past embeddings
            insert_idx = 0
            marker_pos = prompt.find(embedding_insert_marker)
            if marker_pos >= 0:
                left_text = prompt[:marker_pos + len(embedding_insert_marker)]
                left_tokens = self.tokenizer(left_text, add_special_tokens=False)["input_ids"]
                insert_idx = len(left_tokens)
            
            # Split and insert: [left_prompt] + [past_embeddings] + [right_prompt]
            left_emb = input_embeddings[:, :insert_idx, :]
            right_emb = input_embeddings[:, insert_idx:, :]
            combined_input = torch.cat([left_emb, past_embeddings, right_emb], dim=1)
            
            attention_mask = torch.ones(
                (1, combined_input.shape[1]),
                dtype=torch.long,
                device=self.device,
            )
            
            outputs = self.model(
                inputs_embeds=combined_input,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            if not latent_only:
                embedding_record = [combined_input.detach()]
            else:
                embedding_record = []
        
        else:
            # No past context - fresh start
            attention_mask = torch.ones(
                (1, input_embeddings.shape[1]),
                dtype=torch.long,
                device=self.device,
            )
            
            outputs = self.model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            if not latent_only:
                embedding_record = [input_embeddings.detach()]
            else:
                embedding_record = []
        
        generated_tokens: List[int] = []
        
        for step in range(latent_steps):      
            # Apply latent realignment
            latent_vec = self._apply_latent_realignment(last_hidden, self.model, latent_space_realign)
            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, H]
            
            # Record the latent embedding
            embedding_record.append(latent_embed.detach())
            
            # Forward pass with latent embedding
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        # Concatenate all embeddings
        if embedding_record:
            latent_embeddings = torch.cat(embedding_record, dim=1)
        else:
            # latent_only with no latent steps
            latent_embeddings = torch.empty(1, 0, input_embeddings.shape[-1], device=self.device)
        
        return latent_embeddings, latent_steps, generated_tokens, past

    @torch.no_grad()
    def generate_text_with_embeddings_for_api(
        self,
        prompt: str,
        latent_embeddings: torch.Tensor,
        vllm_engine: "LLM",
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        add_think_token: bool = False,
        embedding_insert_marker: str = "<|im_start|>user\n",
    ) -> str:
        """
        Generate text using vLLM with injected latent embeddings.
        
        This method embeds the text prompt, inserts the latent embeddings at the
        appropriate position (after the user message start marker), and generates
        text using vLLM with the combined embeddings.
        
        Args:
            prompt: The text prompt (already formatted with chat template)
            latent_embeddings: Tensor of shape [1, L, H] from generate_latent_with_embeddings_for_api
            vllm_engine: The vLLM LLM engine instance
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            add_think_token: Whether to append <think> token
            embedding_insert_marker: String marker after which to insert latent embeddings
            
        Returns:
            Generated text string
        """
        if add_think_token:
            prompt = f"{prompt}<think>"
        
        # Tokenize the prompt
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        
        # Get the embedding layer
        embedding_layer = self.model.get_input_embeddings()
        
        # Get prompt embeddings
        prompt_embeddings = embedding_layer(input_ids)  # [1, L, H]
        
        
        if latent_embeddings is not None:
            # Find insertion point for latent embeddings
            # Look for the marker in the prompt text
            insert_idx = 0
            marker_pos = prompt.find(embedding_insert_marker)
            if marker_pos >= 0:
                # Tokenize just the text up to and including the marker
                left_text = prompt[:marker_pos + len(embedding_insert_marker)]
                left_tokens = self.tokenizer(left_text, add_special_tokens=False)["input_ids"]
                insert_idx = len(left_tokens)
            # Move latent embeddings to same device as prompt embeddings
            latent_embeddings = latent_embeddings.to(prompt_embeddings.device)
            
            # Split prompt embeddings at insertion point and insert latent embeddings
            left_emb = prompt_embeddings[:, :insert_idx, :]
            right_emb = prompt_embeddings[:, insert_idx:, :]
            
            # Concatenate: [left_prompt] + [latent_embeddings] + [right_prompt]
            # Note: latent_embeddings already contains input embeddings from latent mode,
            # so we only take the latent steps portion (skip the input embeddings part)
            # Actually, for flexibility, we take all of latent_embeddings since the caller
            # can decide what to include
            combined_embeddings = torch.cat([left_emb, latent_embeddings, right_emb], dim=1)
        else:
            combined_embeddings = prompt_embeddings
        # Prepare for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        
        # vLLM expects prompt_embeds in a specific format
        prompt_embeds_list = [
            {"prompt_embeds": combined_embeddings[0]}  # Remove batch dimension
        ]
        
        outputs = vllm_engine.generate(prompt_embeds_list, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip() if outputs else ""
        
        return generated_text

