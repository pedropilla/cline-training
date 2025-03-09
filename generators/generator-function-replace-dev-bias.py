import json
import random
import ollama
import argparse
from tqdm import tqdm
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generator.log"),
        logging.StreamHandler()
    ]
)

# Various transformation types
TRANSFORMATIONS = [
    "Add {hook} hook to the React import",
    "Convert function component to arrow function",
    "Replace {component} with {replacement}",
    "Extract logic into separate {name} function",
    "Convert to {hook} hook pattern",
    "Refactor {method} to use async/await",
    "Add error handling to {function}",
    "Add type annotations to {variable}",
    "Convert class component to functional component",
    "Add prop destructuring in function signature",
    "Replace direct state mutation with setState call",
    "Add useEffect cleanup function",
    "Convert inline styles to styled-components",
    "Add memo wrapper to component",
    "Add ref to {element} element",
    "Replace {library} library with {alternative}",
    "Move state from {parentComponent} to {childComponent}",
    "Convert {event} handler to use useCallback",
    "Add loading state to {component}",
    "Refactor to use context API instead of prop drilling"
]

# Various components and hooks
HOOKS = ["useState", "useEffect", "useContext", "useReducer", "useCallback", "useMemo", "useRef", "useLayoutEffect", "useImperativeHandle", "useDebugValue"]
COMPONENTS = ["Button", "Form", "Modal", "Dropdown", "Card", "Navbar", "Sidebar", "Table", "List", "Input", "Checkbox", "Toggle", "Slider", "Tooltip", "Avatar", "Menu", "Footer", "Header", "Pagination", "SearchBox"]
LIBRARIES = [("axios", "fetch"), ("moment", "date-fns"), ("lodash", "ramda"), ("jQuery", "vanilla JS"), ("Redux", "Context API"), ("styled-components", "emotion"), ("React Router", "Reach Router"), ("Material-UI", "Chakra UI"), ("Formik", "React Hook Form")]
METHODS = ["handleSubmit", "fetchData", "updateState", "validateForm", "processInput", "filterItems", "sortData", "calculateTotal", "renderItems", "toggleVisibility"]
ELEMENT_TYPES = ["button", "input", "div", "span", "form", "img", "a", "p", "h1", "select"]
FILE_TYPES = ["tsx", "jsx", "js", "ts"]

def generate_prompt(index):
    """Generate a random code transformation prompt."""
    transformation_template = random.choice(TRANSFORMATIONS)
    
    # Fill in template placeholders
    transformation = transformation_template
    
    if "{hook}" in transformation:
        transformation = transformation.replace("{hook}", random.choice(HOOKS))
    
    if "{component}" in transformation and "{replacement}" in transformation:
        component = random.choice(COMPONENTS)
        remaining = [c for c in COMPONENTS if c != component]
        replacement = random.choice(remaining)
        transformation = transformation.replace("{component}", component).replace("{replacement}", replacement)
    
    if "{library}" in transformation and "{alternative}" in transformation:
        library, alternative = random.choice(LIBRARIES)
        transformation = transformation.replace("{library}", library).replace("{alternative}", alternative)
    
    if "{method}" in transformation or "{function}" in transformation:
        method = random.choice(METHODS)
        transformation = transformation.replace("{method}", method).replace("{function}", method)
    
    if "{element}" in transformation:
        transformation = transformation.replace("{element}", random.choice(ELEMENT_TYPES))
    
    if "{variable}" in transformation:
        variables = ["props", "state", "data", "items", "users", "config", "options", "settings", "results", "values"]
        transformation = transformation.replace("{variable}", random.choice(variables))
    
    if "{name}" in transformation:
        names = ["helper", "utility", "formatter", "validator", "processor", "handler", "converter", "calculator", "checker", "mapper"]
        transformation = transformation.replace("{name}", random.choice(names))
    
    if "{parentComponent}" in transformation and "{childComponent}" in transformation:
        parent = random.choice(COMPONENTS)
        remaining = [c for c in COMPONENTS if c != parent]
        child = random.choice(remaining)
        transformation = transformation.replace("{parentComponent}", parent).replace("{childComponent}", child)
    
    if "{event}" in transformation:
        events = ["click", "change", "submit", "hover", "focus", "blur", "keypress", "scroll", "resize", "load"]
        transformation = transformation.replace("{event}", random.choice(events))
        
    # Additional instructions
    additional_actions = [
        f"Rename variable {random.choice(['data', 'items', 'result', 'values', 'config', 'options'])} to {random.choice(['newData', 'newItems', 'newResult', 'newValues', 'newConfig', 'newOptions'])}",
        f"Move {random.choice(METHODS)} function {random.choice(['before', 'after'])} the {random.choice(['component declaration', 'import statements', 'interface definition', 'return statement'])}",
        f"Add type annotation to {random.choice(['props', 'state', 'function parameters', 'return value'])}",
        f"Remove unused {random.choice(['imports', 'variables', 'functions', 'comments'])}",
        f"Add a {random.choice(['loading', 'error', 'success'])} state",
        f"Replace anonymous function with named function",
        f"Convert {random.choice(['for loop', 'forEach', 'map'])} to {random.choice(['for loop', 'forEach', 'map', 'reduce', 'filter'])}",
        f"Add {random.choice(['conditional rendering', 'early return', 'null check'])} for {random.choice(['loading state', 'empty data', 'error handling'])}",
        f"Extract {random.choice(['styles', 'logic', 'JSX'])} into separate {random.choice(['file', 'component', 'function'])}",
        f"Add proper indentation to the code"
    ]
    
    # Combine actions
    num_additional = random.randint(0, 2)
    all_actions = [transformation]
    
    for _ in range(num_additional):
        action = random.choice(additional_actions)
        if action not in all_actions:
            all_actions.append(action)
    
    # Build final prompt
    file_path = f"src/components/{random.choice(['App', 'Form', 'Dashboard', 'Profile', 'Settings', 'UserList', 'ProductDetails', 'Login', 'Checkout', 'Navigation'])}.{random.choice(FILE_TYPES)}"
    prompt = f"{'. '.join(all_actions)}. Apply these changes to the file at {file_path}."
    
    return prompt

def generate_ollama_prompt(prompt, model):
    """Create a prompt for Ollama to generate a code transformation example."""
    return f"""
You are an expert code transformation assistant. Given a transformation request, you'll provide a response in a specific format that includes:
1. The original code snippet (you should invent a realistic example)
2. The transformed code with the requested changes

Your response should be formatted as a valid JSON object with a "replace_in_file" command that shows the diff.

Transformation request: "{prompt}"

Please respond with a JSON object in the following format:
{{
  "prompt": "{prompt}",
  "completion": "<replace_in_file>\\n<path>FILE_PATH</path>\\n<diff>\\n<<<<<<< SEARCH\\nORIGINAL_CODE\\n=======\\nTRANSFORMED_CODE\\n>>>>>>> REPLACE\\n</diff>\\n</replace_in_file>\\n"
}}

The diff can include multiple search/replace sections as needed. Be creative but realistic in generating example React/TypeScript code that would benefit from this transformation.
"""

def call_ollama(prompt, model):
    """Call Ollama API to generate a code transformation example."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return None

def parse_ollama_response(response):
    """Parse the Ollama response to extract the JSON object."""
    if not response:
        return None
    
    try:
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            logging.warning("Could not find JSON in response")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate code transformation examples using Ollama')
    parser.add_argument('--count', type=int, default=1000, help='Number of examples to generate')
    parser.add_argument('--output', type=str, default='transformations.jsonl', help='Output file path')
    parser.add_argument('--model', type=str, default='llama3', help='Ollama model to use')
    parser.add_argument('--batch-size', type=int, default=10, help='How many examples to generate per batch')
    args = parser.parse_args()
    
    logging.info(f"Starting generation of {args.count} examples using model {args.model}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful_examples = 0
    batch_num = 0
    
    with open(args.output, 'w') as f:
        pbar = tqdm(total=args.count)
        
        while successful_examples < args.count:
            batch_num += 1
            batch_size = min(args.batch_size, args.count - successful_examples)
            logging.info(f"Starting batch {batch_num} with size {batch_size}")
            
            for i in range(batch_size):
                example_index = successful_examples + i + 1
                prompt = generate_prompt(example_index)
                ollama_prompt = generate_ollama_prompt(prompt, args.model)
                
                response = call_ollama(ollama_prompt, args.model)
                example = parse_ollama_response(response)
                
                if example and "prompt" in example and "completion" in example:
                    try:
                        # Validate that completion contains the expected format
                        completion = example["completion"]
                        if "<replace_in_file>" in completion and "<path>" in completion and "<diff>" in completion:
                            f.write(json.dumps(example) + '\n')
                            f.flush()
                            successful_examples += 1
                            pbar.update(1)
                            
                            if successful_examples >= args.count:
                                break
                        else:
                            logging.warning(f"Example {example_index} has invalid completion format")
                    except Exception as e:
                        logging.error(f"Error writing example {example_index}: {e}")
                else:
                    logging.warning(f"Failed to generate valid example {example_index}")
    
    pbar.close()
    logging.info(f"Generation complete. Generated {successful_examples} examples.")

if __name__ == "__main__":
    main()
