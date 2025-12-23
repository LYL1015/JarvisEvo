import os
import json
import base64
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from openai import OpenAI
import ast

from utils.lrc_tools import LightroomManager
from utils.aigc_tools import AIGCManager
from utils.lua_converter import LuaConverter
from prompts import SYSTEM_PROMPT, REFLECT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT



class Response:
    """Wrapper for API response text"""
    def __init__(self, text):
        self.response_text = text


class APIClient:
    """OpenAI-compatible API client for vision-language models"""
    
    def __init__(self, api_endpoint, api_port, model_name="qwen3_vl", api_key="0", api_timeout=30):
        """
        Initialize API client
        
        Args:
            api_endpoint: API server address
            api_port: API server port
            model_name: Model identifier
            api_key: Authentication key
            api_timeout: API connection timeout in seconds
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_port = api_port
        self.api_timeout = api_timeout
        self.api_connected = False
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=f"http://{api_endpoint}:{api_port}/v1",
                timeout=api_timeout
            )
            self.api_connected = True
        except Exception as e:
            print(f"❌ API client initialization failed: {e}")
            print("⚠️ Program will continue but API functionality unavailable")
            self.client = None
            self.api_connected = False
    
    def chat(self, messages, system=None, images=None, default_timeout=180, **kwargs):
        """
        Send chat request with optional images
        
        Args:
            messages: List of conversation messages
            system: Optional system prompt
            images: Optional list of image paths
            default_timeout: Request timeout in seconds
            **kwargs: Additional API parameters
            
        Returns:
            List containing Response object
        """
        try:
            formatted_messages = self._format_messages(messages, system, images)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=False,
                timeout=default_timeout,
                **kwargs
            )
            return [Response(response.choices[0].message.content)]
        except Exception as e:
            print(f"❌ API call error: {e}")
            return [Response(f"API call failed: {e}")]
        
    def chat_simple(self, formatted_messages, default_timeout=180, **kwargs):
        """
        Send chat request with pre-formatted messages
        
        Args:
            formatted_messages: Pre-formatted message list
            default_timeout: Request timeout in seconds
            **kwargs: Additional API parameters
            
        Returns:
            List containing Response object
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=False,
                timeout=default_timeout,
                **kwargs
            )
            return [Response(response.choices[0].message.content)]
        except Exception as e:
            print(f"❌ API call error: {e}")
            return [Response(f"API call failed: {e}")]
    
    def _format_messages(self, messages, system, images):
        """Format messages with system prompt and images"""
        formatted = []
        
        if system:
            formatted.append({"role": "system", "content": system})
        
        image_idx = 0
        for msg in messages:
            if images and msg["role"] == "user" and image_idx < len(images):
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(images[image_idx])}"}
                    },
                    {"type": "text", "text": msg["content"]}
                ]
                formatted.append({"role": msg["role"], "content": content})
                image_idx += 1
            else:
                formatted.append(msg)
        
        return formatted


# Utility Functions

def encode_image(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compact_text(text):
    """Remove excessive whitespace and line breaks"""
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def extract_tag_content(text, tag):
    """Extract content from XML-style tags"""
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_historical_tool_calls(conversation_history):
    """Extract all tool calls from conversation history"""
    tool_calls = []
    for entry in conversation_history:
        if tool_call := entry.get('tool_call'):
            if tool_call.strip():
                tool_calls.append(tool_call.strip())
    return ' '.join(tool_calls)


# Conversation History Management

class ConversationManager:
    """Manages conversation history storage and retrieval"""
    
    @staticmethod
    def save_round(conversation_history, round_num, messages, full_response, 
                   current_image, tool_call, output_image=None, success=True):
        """
        Save single round of conversation
        
        Args:
            conversation_history: Conversation history list
            round_num: Current round number
            messages: Messages sent to model
            full_response: Complete model response
            current_image: Input image path
            tool_call: Tool call content
            output_image: Output image path
            success: Whether round succeeded
            
        Returns:
            Updated conversation history
        """
        print(f"✅ Saving round {round_num} (total: {len(conversation_history) + 1})")
        
        # Extract thinking content
        thinking = extract_tag_content(full_response, 'think')
        
        # Clean messages (remove image data)
        cleaned_messages = ConversationManager._clean_messages(messages)
        
        round_data = {
            "round": round_num,
            "input_messages": cleaned_messages,
            "full_response": full_response,
            "input_image": current_image,
            "tool_call": tool_call,
            "output_image": output_image,
            "thinking": thinking,
            "success": success
        }
        
        conversation_history.append(round_data)
        return conversation_history
    
    @staticmethod
    def _clean_messages(messages):
        """Remove image data from messages to reduce storage"""
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                continue
    
            content = msg["content"]
            
            # Text-only message
            if isinstance(content, str):
                cleaned.append({"role": msg["role"], "content": content})
            
            # Multimodal message - extract text only
            elif isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_parts:
                    cleaned.append({"role": msg["role"], "content": " ".join(text_parts)})
        
        return cleaned
    
    @staticmethod
    def save_to_file(conversation_history, session_dir):
        """Save complete conversation history to JSON file"""
        try:
            history_file = os.path.join(session_dir, "conversation_history.json")

            # Ensure all data is JSON serializable
            serializable_history = []
            for conv in conversation_history:
                serializable_conv = {
                    k: (v if isinstance(v, (str, list, dict, type(None))) else str(v))
                    for k, v in conv.items()
                }
                serializable_history.append(serializable_conv)
            
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved conversation history: {history_file}")
        except Exception as e:
            print(f"⚠️ Failed to save conversation history: {e}")


# Reflection Mechanism

def reflect_and_improve(conversation_history, chat_model, image_manager, 
                       user_instruction, overall_score, default_timeout=180):
    """
    Reflect on previous attempts and generate improved parameters
    
    Args:
        conversation_history: Previous conversation rounds
        chat_model: API client instance
        image_manager: Image processing manager
        user_instruction: Original user instruction
        overall_score: Quality score from evaluation
        default_timeout: Request timeout in seconds
        
    Returns:
        Dict with reflection response and re-evaluation
    """
    try:
        print("=" * 80)
        print("REFLECTION MODE")
        print("=" * 80)
        
        # Prepare reflection prompt
        original_image = conversation_history[0]["input_image"]
        latest_image = conversation_history[-1]["input_image"]
        latest_thinking = conversation_history[-1]["thinking"]
        
        self_evaluation = {
            "Score": overall_score,
            "Summary": latest_thinking
        }
        
        historical_params = extract_historical_tool_calls(conversation_history)
        
        # Build reflection message
        messages = [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(original_image)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(latest_image)}"}},
                    {
                        "type": "text",
                        "text": (
                            f"1. Original Image: Unedited source image above\n"
                            f"2. Previous Result: Image after previous editing\n"
                            f"3. User Instruction: {user_instruction}\n"
                            f"4. Self-Evaluation: {json.dumps(self_evaluation, ensure_ascii=False)}\n"
                            f"5. Historical Parameters (LUA): {historical_params}\n\n"
                            f"Analyze the gap and provide refined Lightroom adjustments."
                        )
                    }
                ]
            }
        ]
        
        # Get reflection response
        response = chat_model.chat_simple(messages, default_timeout=default_timeout)
        reflection_text = response[0].response_text
        print(f"Reflection: {reflection_text}")

        # Extract improved tool call
        tool_call = extract_tag_content(reflection_text, 'tool_call')
        thinking = extract_tag_content(reflection_text, 'think')

        if tool_call:
            tool_call = compact_text(tool_call)
        
        # Apply reflected parameters
        result_dir = os.path.dirname(latest_image)
        reflected_image_path = os.path.join(result_dir, "reflected_image.jpg")
        
        if tool_call:
            processed_path = image_manager.process_image(original_image, tool_call)
            lua_path = os.path.join(result_dir, "reflected.lua")
            
            # Save Lua preset
            _save_lua_preset(tool_call, lua_path)
        else:
            processed_path = latest_image
            
        # Copy processed image
        if processed_path and os.path.exists(processed_path):
            shutil.copy2(processed_path, reflected_image_path)
        
        # Re-evaluate reflected result
        messages.append({
            "role": "assistant",
            "content": f"<think>{thinking}</think>\n<tool_call>{tool_call}</tool_call>"
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(reflected_image_path)}"}},
                {"type": "text", "text": "Here is the reflected result. Please evaluate again."}
            ]
        })

        re_eval_response = chat_model.chat_simple(messages, default_timeout=default_timeout)[0].response_text

        return {
            "reflectAnswer": reflection_text,
            "answer4evalAgain": re_eval_response
        }

    except Exception as e:
        print(f"❌ Reflection error: {e}")
        return None


def _save_lua_preset(tool_call_json, output_path):
    """Convert JSON tool call to Lua preset file"""
    try:
        json_data = ast.literal_eval(tool_call_json)
        lua_content = 'return ' + LuaConverter.to_lua(json_data)
    except Exception as e:
        print(f"⚠️ Lua conversion failed: {e}")
        lua_content = 'return {}'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(lua_content)


# Main Inference Pipeline

def run_inference(image_path, system_prompt, user_prompt, chat_model, 
                 image_manager, aigc_manager, save_base_path, task_type="lightroom",
                 max_rounds=10, quality_threshold=3.0, default_timeout=180):
    """
    Execute multi-round AI-powered image editing
    
    Args:
        image_path: Path to input image
        system_prompt: System prompt for model
        user_prompt: User editing instruction
        chat_model: API client instance
        image_manager: Lightroom processing manager
        aigc_manager: AIGC processing manager
        save_base_path: Base directory for results
        task_type: Processing mode (lightroom/aigc/auto)
        max_rounds: Maximum number of processing rounds
        quality_threshold: Minimum quality score threshold
        default_timeout: Request timeout in seconds
        
    Returns:
        None (saves results to disk)
    """
    try:
        # Initialize state
        messages = []
        conversation_history = []
        current_image = None
        evaluation_mode = False
        detected_task_type = task_type if task_type != "auto" else None

        # Setup result directories
        result_dir = os.path.join(save_base_path, image_path.split('/')[-2])
        multi_round_dir = os.path.join(result_dir, "MR_image")

        # Skip if already processed
        if os.path.exists(os.path.join(result_dir, "conversation_history.json")):
            print(f"⏭️ Already processed: {image_path}")
            return
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(multi_round_dir, exist_ok=True)
        
        # Copy original image
        if os.path.exists(image_path):
            original_image = os.path.join(result_dir, os.path.basename(image_path))
            shutil.copy2(image_path, original_image)
            current_image = original_image
        
        # Main processing loop
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'=' * 80}")
            print(f"ROUND {round_num}")
            print(f"{'=' * 80}")
                        
            # Build messages for current round
            messages = _build_round_messages(
                messages, conversation_history, round_num, 
                user_prompt, task_type, evaluation_mode, detected_task_type
            )
            
            # Prepare images
            images = _prepare_images(round_num, original_image, current_image)
            
            # Ensure first message is from user (API requirement)
            if messages and messages[0]["role"] != "user":
                messages.insert(0, {"role": "user", "content": f"Analyze image for round {round_num}"})
            
            try:
                # Get model response
                responses = chat_model.chat(
                    messages=messages,
                    system=system_prompt,
                    images=images,
                    default_timeout=default_timeout
                )

                full_response = responses[0].response_text
                
                if not full_response:
                    print("⚠️ Empty response received")
                    continue
                
                # Parse response
                tool_call = extract_tag_content(full_response, 'tool_call')
                state = extract_tag_content(full_response, 'state')
                
                if tool_call:
                    tool_call = compact_text(tool_call)
                
                # Check if evaluation mode
                if state and "finished" in state.lower():
                    evaluation_mode = True
                
                # Detect task type from tool call
                if tool_call and not detected_task_type:
                    if "AIGC-based image editing" in tool_call:
                        detected_task_type = "aigc"
                        tool_call = user_prompt
                    else:
                        detected_task_type = "lightroom"

                # Save round to history
                conversation_history = ConversationManager.save_round(
                    conversation_history, round_num, messages, full_response,
                    current_image, tool_call, success=(tool_call is not None)
                )

                # Check for completion
                if "Overall assessment score" in full_response:
                    print(f"✅ Task completed in round {round_num}")
                    
                    # Extract score and reflect if needed
                    score_match = re.search(r'Overall assessment score\s*:\s*"?\s*(\d+\.?\d*)', full_response)
                    if score_match:
                        score = float(score_match.group(1))
                        if score < quality_threshold:
                            print(f"⚠️ Low score ({score}), triggering reflection")
                            reflection_result = reflect_and_improve(
                                conversation_history, chat_model, image_manager,
                                user_prompt, score, default_timeout
                            )
                            if reflection_result:
                                conversation_history.append(reflection_result)
                    break
                
                # Process image
                processed_image = _process_image(
                    tool_call, detected_task_type, current_image,
                    image_manager, aigc_manager, multi_round_dir,
                    round_num, os.path.basename(image_path)
                )
                
                # Update current image
                if processed_image and os.path.exists(processed_image):
                    current_image = processed_image
                    conversation_history[-1]["output_image"] = processed_image
                
            except Exception as e:
                print(f"❌ Error in round {round_num}: {e}")
                ConversationManager.save_round(
                    conversation_history, round_num, messages, str(e),
                    current_image, None, success=False
                )
                break

        # Save final history
        ConversationManager.save_to_file(conversation_history, result_dir)
    
    except Exception as e:
        print(f"❌ Inference error: {e}")


def _build_round_messages(messages, history, round_num, user_prompt, 
                         task_type, evaluation_mode, detected_task_type):
    """Build message list for current round"""
    if round_num == 1:
        return [{"role": "user", "content": f"<task_type>{task_type}</task_type>. Instruction: {user_prompt}"}]
    
    # Add previous assistant response
    if history:
        prev = history[-1]
        if prev.get("full_response"):
            response_parts = []
            
            response_parts.append(f"<state>{'finished' if evaluation_mode else 'processing'}</state>")
            
            if prev.get("thinking"):
                response_parts.append(f"<think>{prev['thinking']}</think>")
            
            if prev.get("tool_call"):
                response_parts.append(f"<tool_call>{prev['tool_call']}</tool_call>")
            
            messages.append({
                "role": "assistant",
                "content": "\n\n".join(response_parts) if response_parts else f"Round {prev['round']} completed"
            })
    
    # Add current round prompt
    if evaluation_mode:
        if detected_task_type == "lightroom":
            prompt = "Lightroom editing completed. Please evaluate the results."
        elif detected_task_type == "aigc":
            prompt = "AIGC task completed. Please evaluate the results."
        else:
            prompt = "Task completed. Please evaluate the results."
    else:
        prompt = f"Image after step {round_num - 1} of editing."
    
    messages.append({"role": "user", "content": prompt})
    return messages


def _prepare_images(round_num, original_image, current_image):
    """Prepare image list for API call"""
    if round_num == 1:
        return [original_image] if original_image else None
    else:
        return [current_image if current_image else original_image]


def _process_image(tool_call, task_type, current_image, image_manager, 
                  aigc_manager, output_dir, round_num, image_filename):
    """Process image based on tool call and task type"""
    if not tool_call:
        return current_image
    
    output_path = os.path.join(output_dir, f"round_{round_num}_processed_{image_filename}")
    
    # Skip if already exists
    if os.path.exists(output_path):
        return output_path
    
    # Process based on task type
    if task_type == "aigc" and aigc_manager:
        processed = aigc_manager.call_img2img(current_image, tool_call, output_path)
    elif task_type=="lightroom" and image_manager:
        processed = image_manager.process_image(current_image, tool_call)
        
        # Save Lua preset
        lua_path = os.path.join(output_dir, f"round_{round_num}_processing.lua")
        _save_lua_preset(tool_call, lua_path)
    
    # Copy to output path
    if processed and os.path.exists(processed):
        shutil.copy2(processed, output_path)
        return output_path
    
    return current_image


# Batch Processing

def process_single_image(path, image_base_path, chat_model, image_manager, 
                        aigc_manager, system_prompt, save_base_path, 
                        prompt_file_name, task_type="lightroom",
                        max_rounds=10, quality_threshold=3.0, default_timeout=180):
    """
    Process single image with AI editing
    
    Args:
        path: Relative path to image directory
        image_base_path: Base directory containing images
        chat_model: API client instance
        image_manager: Lightroom manager
        aigc_manager: AIGC manager
        system_prompt: System prompt
        save_base_path: Results directory
        prompt_file_name: User prompt filename
        task_type: Processing mode
        max_rounds: Maximum number of processing rounds
        quality_threshold: Minimum quality score threshold
        default_timeout: Request timeout in seconds
        
    Returns:
        Status message
    """
    try:
        base_path = os.path.join(image_base_path, path)

        # Skip if already processed
        if os.path.exists(os.path.join(base_path, "conversation_history.json")):
            return f"⏭️ Skipped (already processed): {path}"
        
        # Find image file
        image_path = os.path.join(base_path, "before.jpg")
        if not os.path.exists(image_path):
            for temp_in_Name in ["before.png", "input.jpg", "input.png"]:
                image_path = os.path.join(base_path, temp_in_Name)
                if os.path.exists(image_path):
                    break
        
        if not os.path.exists(image_path):
            return f"⚠️ Skipped (no image): {path}"
        
        # Read user prompt
        prompt_path = os.path.join(base_path, prompt_file_name)
        if not os.path.isfile(prompt_path):
            return f"⚠️ Skipped (no prompt): {path}"
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                user_prompt = f.read()
        except Exception as e:
            return f"❌ Error reading prompt: {path} - {e}"
        
        # Process image
        thread_id = threading.current_thread().ident
        print(f"[Thread {thread_id}] Processing: {image_path}")
        
        run_inference(
            image_path, system_prompt, user_prompt, chat_model,
            image_manager, aigc_manager, save_base_path, task_type,
            max_rounds, quality_threshold, default_timeout
        )
        
        print(f"[Thread {thread_id}] ✅ Completed: {path}")
        return f"✅ Completed: {path}"
        
    except Exception as e:
        thread_id = threading.current_thread().ident
        error_msg = f"[Thread {thread_id}] ❌ Error: {path} - {e}"
        print(error_msg)
        return error_msg


# Main Entry Point

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JarvisEvo Inference - Multi-modal AI image editing"
    )
    
    # API Configuration
    parser.add_argument("--api_endpoint", type=str, default="localhost", 
                       help="API server address")
    parser.add_argument("--api_port", type=int, nargs='+', default=[8086], 
                       help="API server port(s) for load balancing")
    parser.add_argument("--api_key", type=str, default="0", 
                       help="API authentication key")
    parser.add_argument("--model_name", type=str, default="qwen3_vl", 
                       help="AI model name")
    
    # Processing Configuration
    parser.add_argument("--max_threads", type=int, default=10, 
                       help="Maximum concurrent threads")
    parser.add_argument("--task_type", type=str, default="lightroom", 
                       help="Processing mode: lightroom/aigc/auto")
    
    # File Paths
    parser.add_argument("--image_path", type=str, default=None,
                       help="Input image directory")
    parser.add_argument("--save_base_path", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--prompt_file_name", type=str, default="user_want.txt", 
                       help="User prompt filename")
    
    # AIGC Configuration
    parser.add_argument("--AIGC_model_pth", type=str, default=None, 
                       help="AIGC model path")
    parser.add_argument("--AIGC_device", type=str, default="cuda:0", 
                       help="AIGC device")
    
    # Processing Parameters
    parser.add_argument("--max_rounds", type=int, default=10,
                       help="Maximum number of processing rounds")
    parser.add_argument("--quality_threshold", type=float, default=3.0,
                       help="Minimum quality score threshold for reflection")
    parser.add_argument("--default_timeout", type=int, default=180,
                       help="Default timeout for API requests in seconds")
    parser.add_argument("--api_timeout", type=int, default=30,
                       help="API connection timeout in seconds")
    
    args = parser.parse_args()
    
    # Initialize API clients
    api_ports = args.api_port if isinstance(args.api_port, list) else [args.api_port]
    chat_models = [
        APIClient(args.api_endpoint, port, args.model_name, args.api_key, args.api_timeout)
        for port in api_ports
    ]
    
    # Initialize managers
    image_manager = LightroomManager() 
    if args.AIGC_model_pth:
        aigc_manager = AIGCManager(args.AIGC_model_pth, args.AIGC_device)
    else:
        aigc_manager = None

    # Get image list
    image_dirs = sorted(os.listdir(args.image_path))
    print(f"Processing {len(image_dirs)} images with {args.max_threads} threads")
    
    # Process images concurrently
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(
                process_single_image, 
                path, 
                args.image_path, 
                chat_models[idx % len(chat_models)],  # Load balancing
                image_manager, 
                aigc_manager,
                SYSTEM_PROMPT, 
                args.save_base_path,
                args.prompt_file_name,
                args.task_type,
                args.max_rounds,
                args.quality_threshold,
                args.default_timeout
            ): path
            for idx, path in enumerate(image_dirs)
        }
        
        # Monitor progress
        with tqdm.tqdm(total=len(image_dirs), desc="Processing") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    print(f"\n{result}")
                except Exception as e:
                    print(f"\n❌ Exception for {path}: {e}")
                finally:
                    pbar.update(1)
    
    print(f"\n✅ Processing complete: {len(image_dirs)} images")


if __name__ == "__main__":
    main()
