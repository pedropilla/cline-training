import os
import json
import re
import argparse
import requests
import time
import uuid
import random
from typing import List, Dict, Tuple, Optional, Literal
from datetime import datetime

class LLMInterface:
    """Base class for LLM interfaces"""
    def query(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_system_prompt(self) -> str:
        """Return the system prompt to guide the LLM"""
        return """
        You are an expert instruction analyzer specialized in extracting high-quality question-answer pairs from technical guidelines.
        
        Your purpose is to:
        1. Carefully analyze the provided instruction text
        2. Extract meaningful questions that users might ask about these instructions
        3. Provide precise answers derived directly from the instruction text
        4. Format your output as valid JSON objects with "prompt" and "completion" fields
        
        Guidelines for extraction:
        - Extract exact phrasing from the text when possible for the answers
        - Create diverse question types (what/how/when/why questions)
        - Ensure answers are factual and directly supported by the instruction text
        - Avoid making assumptions beyond what's explicitly stated
        - Identify rules, guidelines, formatting requirements, and procedural steps
        - Focus on actionable information that clarifies how to follow the instructions
        
        Important: Your extracted Q&A pairs will be used to train an AI assistant to properly follow these instructions, so precision and accuracy are essential.
        
        Return your response as properly formatted JSON objects, one per line, with each having:
        {"prompt": "Question about the instruction", "completion": "Answer from the instruction text"}
        """

class OllamaInterface(LLMInterface):
    """Interface for Ollama API"""
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama3"):
        self.url = f"http://{host}:{port}/api/chat"  # Changed to chat endpoint
        self.model = model
        
    def query(self, prompt: str) -> str:
        system_prompt = self.get_system_prompt()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        try:
            # Add small delay to prevent overwhelming the API
            time.sleep(0.5)
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            # Fallback to generate API if chat API fails
            return self.fallback_query(prompt, system_prompt)
    
    def fallback_query(self, prompt: str, system_prompt: str) -> str:
        """Fallback to generate API if chat API is not available"""
        fallback_url = f"http://{self.url.split('/')[2]}/api/generate"
        combined_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
        
        payload = {
            "model": self.model,
            "prompt": combined_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(fallback_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error in fallback query: {e}")
            return ""

class LMStudioInterface(LLMInterface):
    """Interface for LM Studio API"""
    def __init__(self, host: str = "localhost", port: int = 1234, model: str = "qwq-32b"):
        self.url = f"http://{host}:{port}/v1/chat/completions"  # Updated to chat completions endpoint
        self.model = model
        
    def query(self, prompt: str) -> str:
        system_prompt = self.get_system_prompt()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800,  # Increased token limit for more complete responses
            "stream": False
        }
        
        try:
            # Add small delay to prevent overwhelming the API
            time.sleep(0.5)
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Error querying LM Studio: {e}")
            # Fallback to completions API if chat API fails
            return self.fallback_query(prompt, system_prompt)
    
    def fallback_query(self, prompt: str, system_prompt: str) -> str:
        """Fallback to completions API if chat API is not available"""
        fallback_url = f"http://{self.url.split('/')[2]}/v1/completions"
        combined_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
        
        payload = {
            "model": self.model,
            "prompt": combined_prompt,
            "temperature": 0.7,
            "max_tokens": 800,
            "stop": ["<|end|>", "<|user|>"],
            "stream": False
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(fallback_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("text", "")
        except requests.exceptions.RequestException as e:
            print(f"Error in fallback query: {e}")
            return ""

def create_llm_interface(
    backend: Literal["ollama", "lmstudio"], 
    host: str, 
    port: int, 
    model: Optional[str] = None
) -> LLMInterface:
    """Factory function to create the appropriate LLM interface"""
    if backend == "ollama":
        return OllamaInterface(host, port, model or "llama3")
    elif backend == "lmstudio":
        return LMStudioInterface(host, port, model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def generate_extraction_prompt(instruction: str, iteration: int) -> str:
    """
    Generate a prompt for extracting Q&A pairs based on the iteration.
    
    Args:
        instruction: The instruction text to analyze
        iteration: Current iteration number to vary the approach
        
    Returns:
        A prompt string for the LLM
    """
    # Check for special patterns in the instruction to customize extraction
    has_numbered_items = bool(re.search(r'\d+\.\s+', instruction))
    has_xml_examples = '<' in instruction and '>' in instruction
    has_guidelines = 'guidelines' in instruction.lower()
    has_rules = 'rules' in instruction.lower()
    has_definitions = ':' in instruction and not instruction.strip().endswith(':')
    
    # Base prompt variations
    base_prompts = [
        # General extraction
        f"""
        Given this instruction text: "{instruction}"
        
        Extract 10 different user questions that could be asked about this instruction.
        For each question, provide the precise answer from the instruction.
        Format each as a valid JSON object with "prompt" (question) and "completion" (answer) fields.
        Return only valid JSON objects, one per line (JSONL format).
        Be creative and extract different types of questions than previous iterations.
        """,
        
        # Scenario-based questions
        f"""
        From this instruction: "{instruction}"
        
        Create 10 scenario-based questions like "What should I do if..." or "How do I..." 
        that relate to the instruction, with accurate answers.
        Format as JSONL with "prompt" and "completion" fields.
        Try to extract different aspects than previous responses.
        """,
        
        # Terminology focus
        f"""
        Analyze this instruction: "{instruction}"
        
        Identify 10 key terms, concepts or modes mentioned and create question-answer pairs
        that explain what each one means or does.
        Format as JSONL with "prompt" and "completion" fields.
        Find aspects not covered in previous iterations.
        """,
        
        # Constraints and limitations
        f"""
        From this instruction text: "{instruction}"
        
        Extract 10 questions about limitations, constraints, or exceptions mentioned.
        Format as JSONL with "prompt" (e.g., "What can't I do in X mode?") and "completion" fields.
        Focus on aspects not previously covered.
        """,
        
        # Procedural questions
        f"""
        Given this instruction: "{instruction}"
        
        Create 10 procedural questions like "How do I accomplish X?" or "What's the process for Y?"
        that could be answered based on the instruction.
        Format as JSONL with "prompt" and "completion" fields.
        Find new angles not covered in previous extractions.
        """
    ]
    
    # Specialized prompts for patterns
    specialized_prompts = []
    
    if has_numbered_items:
        specialized_prompts.append(f"""
        This instruction contains numbered items: "{instruction}"
        
        For each numbered item (1., 2., etc.), create separate question-answer pairs.
        Format questions like "What is step X?" or "Explain guideline #Y" with the exact content as answers.
        Also create questions that reference specific details within each numbered item.
        Return as JSONL with "prompt" and "completion" fields.
        """)
    
    if has_xml_examples:
        specialized_prompts.append(f"""
        This instruction contains XML or code examples: "{instruction}"
        
        Create questions about:
        1. How to format specific commands/tools
        2. What each part of the syntax means
        3. When to use specific formats
        4. How to structure parameters
        5. What errors to avoid in formatting
        
        Include the exact examples in your answers when relevant.
        Return as JSONL with "prompt" and "completion" fields.
        """)
    
    if has_guidelines or has_rules:
        specialized_prompts.append(f"""
        This text contains rules or guidelines: "{instruction}"
        
        For each rule or guideline, create questions like:
        1. "What is the rule about X?"
        2. "Why should I follow guideline Y?"
        3. "What happens if I don't follow rule Z?"
        4. "Is there an exception to guideline W?"
        
        Provide precise answers from the text.
        Return as JSONL with "prompt" and "completion" fields.
        """)
    
    if has_definitions:
        specialized_prompts.append(f"""
        This text contains term definitions: "{instruction}"
        
        For each term defined (usually before a colon):
        1. Create direct questions like "What is X?" or "Define Y"
        2. Create usage questions like "When would I use X?" or "How does Y work?"
        3. Create comparison questions if multiple terms exist
        
        Return as JSONL with "prompt" and "completion" fields.
        """)
    
    # Combine base and specialized prompts
    all_prompts = base_prompts + specialized_prompts
    
    # Use modulo to cycle through prompts, but occasionally use specialized prompts
    if specialized_prompts and (iteration % 8 >= 5 or random.random() < 0.3):
        return random.choice(specialized_prompts)
    else:
        return all_prompts[iteration % len(base_prompts)]

def clean_and_validate_json(json_str: str) -> List[Dict]:
    """
    Clean the LLM output and convert it to valid JSON objects.
    
    Args:
        json_str: String containing potential JSON objects
        
    Returns:
        List of valid JSON objects
    """
    # Find all JSON-like structures in the text
    json_pattern = r'(\{\s*"prompt"\s*:\s*".*?"\s*,\s*"completion"\s*:\s*".*?"\s*\})'
    potential_jsons = re.findall(json_pattern, json_str, re.DOTALL)
    
    # Also try to find JSON objects that might be surrounded by backticks
    if not potential_jsons:
        backtick_pattern = r'```(?:json)?\s*(.+?)```'
        backtick_matches = re.findall(backtick_pattern, json_str, re.DOTALL)
        for match in backtick_matches:
            potential_jsons.extend(re.findall(json_pattern, match, re.DOTALL))
    
    # Try to find numbered JSON objects (1. {...})
    if not potential_jsons:
        numbered_pattern = r'\d+\.\s*(\{\s*"prompt"\s*:\s*".*?"\s*,\s*"completion"\s*:\s*".*?"\s*\})'
        potential_jsons.extend(re.findall(numbered_pattern, json_str, re.DOTALL))
    
    valid_jsons = []
    for json_text in potential_jsons:
        try:
            # Replace escaped quotes and normalize format
            json_text = json_text.replace('\\"', '"')
            
            # Handle nested quotes in JSON values
            json_text = re.sub(r'("prompt"\s*:\s*")(.+?)(")', 
                              lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3), 
                              json_text)
            json_text = re.sub(r'("completion"\s*:\s*")(.+?)(")', 
                              lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3), 
                              json_text)
            
            # Parse JSON
            json_obj = json.loads(json_text)
            
            # Validate that it has the required fields
            if "prompt" in json_obj and "completion" in json_obj:
                # Clean up whitespace and newlines in the fields
                json_obj["prompt"] = json_obj["prompt"].strip()
                json_obj["completion"] = json_obj["completion"].strip()
                valid_jsons.append(json_obj)
        except json.JSONDecodeError:
            # Try to fix common formatting issues
            try:
                # Try with single quotes converted to double quotes
                fixed_text = json_text.replace("'", '"')
                json_obj = json.loads(fixed_text)
                if "prompt" in json_obj and "completion" in json_obj:
                    json_obj["prompt"] = json_obj["prompt"].strip()
                    json_obj["completion"] = json_obj["completion"].strip()
                    valid_jsons.append(json_obj)
            except:
                continue
    
    return valid_jsons

def extract_pairs_manually(instruction: str) -> List[Dict]:
    """
    Extract pairs using regex patterns specialized for the tool use guidelines.
    
    Args:
        instruction: The instruction text
        
    Returns:
        List of extracted Q&A pairs
    """
    pairs = []
    
    # Common patterns in instructions
    # Tool use pattern - e.g., "Tool Use: You have access to a set of tools..."
    tool_use_pattern = r"Tool Use:?\s+(.+?)(?=\n|$)"
    
    # Tool use formatting pattern - e.g., "Tool Use Formatting: \n\nTool use is formatted using..."
    tool_format_pattern = r"Tool Use Formatting:?\s+(.+?)(?=\n\n|$)"
    
    # Numbered guidelines pattern - e.g., "Tool Use Guidelines: 1. In <thinking> tags..."
    numbered_guideline_pattern = r"Tool Use Guidelines:?\s+(\d+)\.\s+(.+?)(?=\n|$)"
    
    # General guidelines pattern - e.g., "Tool Use Guidelines: It is crucial to proceed step-by-step..."
    general_guideline_pattern = r"Tool Use Guidelines:?\s+(?!\d)(.+?)(?=\n\n|$)"
    
    # XML format pattern - capture XML examples
    xml_pattern = r"<([^>]+)>\s*(?:<([^>]+)>([^<]+)</\2>\s*)*</\1>"
    
    # Extract tool use information
    for match in re.findall(tool_use_pattern, instruction, re.DOTALL):
        description = match.strip()
        pairs.append({
            "prompt": "What is Tool Use?",
            "completion": description
        })
        pairs.append({
            "prompt": "How do I use tools?",
            "completion": description
        })
        
        # Extract specific details if present
        if "one tool per message" in description.lower():
            pairs.append({
                "prompt": "How many tools can I use in a single message?",
                "completion": "You can use one tool per message."
            })
        
        if "user's approval" in description.lower():
            pairs.append({
                "prompt": "Do tools execute automatically?",
                "completion": "No, tools are executed upon the user's approval."
            })
            
        if "step-by-step" in description.lower():
            pairs.append({
                "prompt": "What is the approach for using tools to accomplish a task?",
                "completion": "You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use."
            })
    
    # Extract tool formatting information
    for match in re.findall(tool_format_pattern, instruction, re.DOTALL):
        format_info = match.strip()
        pairs.append({
            "prompt": "How should I format tool use?",
            "completion": format_info
        })
        pairs.append({
            "prompt": "What is the format for tool use?",
            "completion": format_info
        })
        
        # Extract XML examples if present
        xml_examples = re.findall(xml_pattern, format_info)
        for example in xml_examples:
            xml_text = re.search(r'(<' + example[0] + '>.*?</' + example[0] + '>)', format_info, re.DOTALL)
            if xml_text:
                pairs.append({
                    "prompt": f"Show an example of the {example[0]} tool format",
                    "completion": xml_text.group(1)
                })
                pairs.append({
                    "prompt": f"How do I use the {example[0]} tool?",
                    "completion": xml_text.group(1)
                })
    
    # Extract numbered guidelines
    for match in re.findall(numbered_guideline_pattern, instruction, re.DOTALL):
        number, guideline = match
        pairs.append({
            "prompt": f"What is guideline #{number} for tool use?",
            "completion": guideline.strip()
        })
        pairs.append({
            "prompt": f"Explain tool use guideline {number}",
            "completion": guideline.strip()
        })
        
        # Create specific questions based on the guideline content
        if "assess" in guideline.lower():
            pairs.append({
                "prompt": "What should I assess before using a tool?",
                "completion": guideline.strip()
            })
        
        if "appropriate tool" in guideline.lower():
            pairs.append({
                "prompt": "How do I choose which tool to use?",
                "completion": guideline.strip()
            })
            
        if "one tool at a time" in guideline.lower():
            pairs.append({
                "prompt": "Can I use multiple tools at once?",
                "completion": "No. If multiple actions are needed, use one tool at a time per message to accomplish the task iteratively."
            })
            
        if "xml format" in guideline.lower():
            pairs.append({
                "prompt": "What format should I use for tool calls?",
                "completion": "Formulate your tool use using the XML format specified for each tool."
            })
            
        if "wait for user confirmation" in guideline.lower():
            pairs.append({
                "prompt": "Should I wait for the user after using a tool?",
                "completion": "Yes, ALWAYS wait for user confirmation after each tool use before proceeding."
            })
    
    # Extract general guidelines
    for match in re.findall(general_guideline_pattern, instruction, re.DOTALL):
        guideline = match.strip()
        pairs.append({
            "prompt": "What are the general guidelines for tool use?",
            "completion": guideline
        })
        
        # Extract specific aspects based on content
        if "step-by-step" in guideline.lower():
            pairs.append({
                "prompt": "Why is a step-by-step approach important for tool use?",
                "completion": guideline
            })
            
        if "confirm the success" in guideline.lower():
            pairs.append({
                "prompt": "Why should I wait for user confirmation between tool uses?",
                "completion": guideline
            })
            
        # Look for numbered points within guidelines
        numbered_points = re.findall(r'(\d+)\.\s+([^.]+)', guideline)
        if numbered_points:
            for num, point in numbered_points:
                pairs.append({
                    "prompt": f"What is benefit #{num} of the step-by-step approach?",
                    "completion": point.strip()
                })
    
    return pairs

def process_instruction(
    instruction: str, 
    max_iterations: int, 
    llm_interface: LLMInterface
) -> List[Dict]:
    """
    Process a single instruction to extract Q&A pairs with multiple iterations.
    
    Args:
        instruction: The instruction text
        max_iterations: Maximum number of iterations to run
        llm_interface: Interface to the language model
        
    Returns:
        List of extracted Q&A pairs
    """
    all_pairs = []
    seen_questions = set()
    seen_answers = set()
    
    # First try manual extraction for reliable basic patterns
    manual_pairs = extract_pairs_manually(instruction)
    for pair in manual_pairs:
        question_key = pair["prompt"].lower()
        answer_key = pair["completion"].lower()
        
        if question_key not in seen_questions and answer_key not in seen_answers:
            all_pairs.append(pair)
            seen_questions.add(question_key)
            seen_answers.add(answer_key)
    
    # Then use LLM for more creative extraction
    iteration_count = 0
    stagnant_counter = 0
    previous_count = len(all_pairs)
    
    print(f"Starting with {previous_count} manually extracted pairs")
    
    while iteration_count < max_iterations and stagnant_counter < 5:
        print(f"  Iteration {iteration_count+1}/{max_iterations} (Stagnant: {stagnant_counter})")
        
        # Generate prompt based on iteration
        prompt = generate_extraction_prompt(instruction, iteration_count)
        
        # Query LLM
        llm_response = llm_interface.query(prompt)
        
        # Extract and validate JSON objects
        pairs = clean_and_validate_json(llm_response)
        
        new_pairs_added = 0
        for pair in pairs:
            question_key = pair["prompt"].lower()
            answer_key = pair["completion"].lower()
            
            # Filter out very short or empty answers
            if len(answer_key) < 5:
                continue
                
            # Skip if we've seen this exact question before
            if question_key in seen_questions:
                continue
                
            # Add to our collection
            all_pairs.append(pair)
            seen_questions.add(question_key)
            seen_answers.add(answer_key)
            new_pairs_added += 1
        
        current_count = len(all_pairs)
        print(f"    Found {len(pairs)} pairs, added {new_pairs_added} new ones (Total: {current_count})")
        
        # Check if we're still finding new pairs
        if current_count == previous_count:
            stagnant_counter += 1
        else:
            stagnant_counter = 0
            
        previous_count = current_count
        iteration_count += 1
        
        # Add small randomization to iteration count to prevent getting stuck
        if random.random() < 0.1:
            iteration_count += random.randint(1, 3)
    
    return all_pairs

def process_file(
    file_path: str, 
    max_iterations_per_line: int, 
    llm_interface: LLMInterface
) -> List[Dict]:
    """
    Process each line of a file as a separate instruction set.
    
    Args:
        file_path: Path to the instruction file
        max_iterations_per_line: Maximum iterations per instruction line
        llm_interface: Interface to the language model
        
    Returns:
        List of all extracted Q&A pairs
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    all_pairs = []
    for i, line in enumerate(lines):
        print(f"\nProcessing line {i+1}/{len(lines)}")
        print(f"Instruction: {line[:100]}...")  # Print first 100 chars
        
        pairs = process_instruction(line, max_iterations_per_line, llm_interface)
        all_pairs.extend(pairs)
        print(f"Extracted {len(pairs)} unique pairs from line {i+1}")
        
        # Periodically save intermediate results
        if (i+1) % 5 == 0 or (i+1) == len(lines):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_file = f"intermediate_results_{timestamp}_{i+1}.jsonl"
            save_to_jsonl(all_pairs, intermediate_file)
            print(f"Saved intermediate results to {intermediate_file}")
    
    return all_pairs

def save_to_jsonl(pairs: List[Dict], output_path: str):
    """
    Save extracted pairs to a JSONL file.
    
    Args:
        pairs: List of Q&A pairs
        output_path: Path to save the JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

def generate_unique_filename(base_filename: str) -> str:
    """
    Generate a unique filename by adding timestamp and random string.
    
    Args:
        base_filename: Base filename to use
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_str = str(uuid.uuid4())[:8]
    
    base_name, extension = os.path.splitext(base_filename)
    
    return f"{base_name}_{timestamp}_{random_str}{extension}"

def main():
    parser = argparse.ArgumentParser(description="Extract Q&A pairs from instruction files for LLM fine-tuning")
    parser.add_argument("input_file", help="Path to the instruction file")
    parser.add_argument("--output", default="output.jsonl", help="Base name for the JSONL output")
    parser.add_argument("--backend", choices=["ollama", "lmstudio"], default="ollama", 
                        help="LLM backend to use (ollama or lmstudio)")
    parser.add_argument("--host", default="localhost", help="Host for LLM backend")
    parser.add_argument("--port", type=int, help="Port for LLM backend (default: 11434 for Ollama, 1234 for LM Studio)", 
                        default=None)
    parser.add_argument("--model", help="Model name (for Ollama only)", default="llama3")
    parser.add_argument("--max-iterations", type=int, default=25, 
                       help="Maximum iterations per instruction line (default: 25)")
    
    args = parser.parse_args()
    
    # Set default port based on backend if not specified
    if args.port is None:
        args.port = 11434 if args.backend == "ollama" else 1234
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Generate unique output filename
    unique_output = generate_unique_filename(args.output)
    
    # Create LLM interface
    llm_interface = create_llm_interface(args.backend, args.host, args.port, args.model)
    
    print(f"Processing file: {args.input_file}")
    print(f"Using backend: {args.backend}")
    if args.backend == "ollama":
        print(f"Using model: {args.model}")
    print(f"Max iterations per line: {args.max_iterations}")
    print(f"Output will be saved to: {unique_output}")
    
    start_time = time.time()
    pairs = process_file(args.input_file, args.max_iterations, llm_interface)
    end_time = time.time()
    
    # Deduplicate pairs to ensure quality
    unique_pairs = []
    seen_questions = set()
    
    for pair in pairs:
        question = pair["prompt"].lower()
        if question not in seen_questions:
            unique_pairs.append(pair)
            seen_questions.add(question)
    
    save_to_jsonl(unique_pairs, unique_output)
    
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Saved {len(unique_pairs)} unique Q&A pairs to {unique_output}")
    
    # Calculate statistics
    lines_processed = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines_processed = sum(1 for line in f if line.strip())
    
    print(f"\nStatistics:")
    print(f"Lines processed: {lines_processed}")
    print(f"Average pairs per line: {len(unique_pairs)/lines_processed:.2f}")
    print(f"Processing time per line: {processing_time/lines_processed:.2f} seconds")
    print(f"Duplicate questions filtered: {len(pairs) - len(unique_pairs)}")

if __name__ == "__main__":
    main()