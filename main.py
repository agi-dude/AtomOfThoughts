import argparse
import json
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
import ollama
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import cmd
import os

class AOTReasoner:
    """
    Implements the Atom of Thoughts (AOT) reasoning technique.
    This technique decomposes questions into a directed acyclic graph (DAG),
    contracts subquestions into simpler questions, and iterates until reaching atomic questions.
    """
    
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the AOT reasoner with a specific LLM model.
        
        Args:
            model_name: The name of the Ollama model to use
        """
        self.model = model_name
        
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response using the Ollama model."""
        try:
            if system_prompt:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {e}"
    
    def decompose_question(self, question: str) -> nx.DiGraph:
        """
        Decompose a question into a directed acyclic graph (DAG) of subquestions.
        
        Args:
            question: The initial question to decompose
            
        Returns:
            A directed acyclic graph representing question dependencies
        """
        decompose_prompt = f"""
        Decompose the following question into subquestions that form a dependency graph:
        
        Question: {question}
        
        Return the result in the following JSON format:
        {{
            "subquestions": [
                {{"id": "q1", "text": "First subquestion"}},
                {{"id": "q2", "text": "Second subquestion"}},
                ...
            ],
            "dependencies": [
                {{"from": "q1", "to": "q2"}},
                ...
            ]
        }}
        
        Note: A dependency means that answering the "from" question requires the answer to the "to" question.
        """
        
        response = self.generate_response(decompose_prompt)
        
        # Extract the JSON part of the response
        try:
            # Find JSON content in the response (it might be wrapped in markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes (subquestions)
            for subq in data["subquestions"]:
                G.add_node(subq["id"], text=subq["text"])
            
            # Add edges (dependencies)
            for dep in data["dependencies"]:
                G.add_edge(dep["from"], dep["to"])
            
            return G
        except Exception as e:
            print(f"Error parsing decomposition: {e}")
            print(f"Raw response: {response}")
            # Create a fallback graph with just the original question
            G = nx.DiGraph()
            G.add_node("q0", text=question)
            return G
    
    def get_max_path_length(self, G: nx.DiGraph) -> int:
        """Calculate the maximum path length in the graph."""
        if len(G.nodes()) == 0:
            return 0
        
        try:
            # Find the longest path
            longest_path_length = 0
            for node in G.nodes():
                # If node has no successors, calculate path length to it
                if G.out_degree(node) == 0:
                    for source in G.nodes():
                        if source != node and nx.has_path(G, source, node):
                            path_length = len(nx.shortest_path(G, source, node)) - 1
                            longest_path_length = max(longest_path_length, path_length)
            
            return longest_path_length + 1  # +1 to account for the final node
        except Exception as e:
            print(f"Error calculating max path length: {e}")
            return 1
    
    def identify_independent_dependent_questions(self, G: nx.DiGraph) -> Tuple[Set[str], Set[str]]:
        """
        Identify independent and dependent questions in the graph.
        
        Independent questions have no incoming edges.
        Dependent questions have at least one incoming edge.
        """
        independent = set()
        dependent = set()
        
        for node in G.nodes():
            if G.in_degree(node) == 0:
                independent.add(node)
            else:
                dependent.add(node)
        
        return independent, dependent
    
    def contract_questions(self, G: nx.DiGraph, independent: Set[str], dependent: Set[str]) -> str:
        """
        Contract subquestions into a simpler question.
        
        Args:
            G: The question dependency graph
            independent: Set of independent question IDs
            dependent: Set of dependent question IDs
            
        Returns:
            A new contracted question
        """
        # Extract the text of all questions
        independent_texts = [G.nodes[q]["text"] for q in independent]
        dependent_texts = [G.nodes[q]["text"] for q in dependent]
        
        # Prepare prompt for contraction
        contract_prompt = f"""
        I have decomposed a complex question into independent subquestions and dependent subquestions.
        
        Independent subquestions (these don't depend on other questions):
        {', '.join(independent_texts)}
        
        Dependent subquestions (these depend on answers to other questions):
        {', '.join(dependent_texts)}
        
        Create a NEW, SIMPLER question that combines these subquestions in a way that preserves 
        the essence of what we're trying to understand. The new question should be more focused
        and eliminate redundancies.
        
        Return ONLY the new question text with no additional explanation.
        """
        
        contracted_question = self.generate_response(contract_prompt)
        return contracted_question.strip()
    
    def solve_final_question(self, question: str, original_question: str) -> str:
        """Solve the final atomic question."""
        solve_prompt = f"""
        Original question: {original_question}
        
        After decomposing and contracting the original question through the Atom of Thoughts process,
        I've arrived at this atomic question:
        
        {question}
        
        Please provide a comprehensive and accurate answer to both this atomic question
        and the original question.
        """
        
        answer = self.generate_response(solve_prompt)
        return answer
    
    def reason(self, question: str, max_iterations: int = 5) -> str:
        """
        Apply the Atom of Thoughts (AOT) reasoning process to answer a question.
        
        Args:
            question: The initial question
            max_iterations: Maximum number of iterations to perform
            
        Returns:
            The final answer
        """
        print(f"Starting AOT reasoning process for: {question}")
        
        # Initialize
        iteration = 0
        D = None  # max depth
        current_question = question
        original_question = question
        
        # Main loop
        while (D is None or iteration < D) and iteration < max_iterations:
            print(f"\nIteration {iteration}")
            print(f"Current question: {current_question}")
            
            # Step 4: Decompose question into DAG
            G = self.decompose_question(current_question)
            
            # Print the graph structure
            print(f"Generated graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
            for node, data in G.nodes(data=True):
                print(f"  Node {node}: {data['text']}")
            for edge in G.edges():
                print(f"  Edge: {edge[0]} -> {edge[1]}")
            
            # Steps 5-7: If D is None, calculate max path length
            if D is None:
                D = self.get_max_path_length(G)
                print(f"Calculated max depth D = {D}")
            
            # Steps 8-9: Identify independent and dependent questions
            Q_ind, Q_dep = self.identify_independent_dependent_questions(G)
            print(f"Independent questions: {Q_ind}")
            print(f"Dependent questions: {Q_dep}")
            
            # Step 10: Contract subquestions into a new question
            if Q_ind and Q_dep:
                next_question = self.contract_questions(G, Q_ind, Q_dep)
            elif len(G.nodes()) > 0:
                # If we have only independent or only dependent questions, use all
                all_nodes = set(G.nodes())
                next_question = self.contract_questions(G, all_nodes, set())
            else:
                # Fallback if no nodes
                next_question = current_question
            
            print(f"Contracted question: {next_question}")
            
            # Step 11: Increment iteration counter
            iteration += 1
            current_question = next_question
        
        # Step 13: Generate final answer
        print(f"\nGenerating final answer for: {current_question}")
        answer = self.solve_final_question(current_question, original_question)
        
        return answer

class AOTChat:
    """Multi-turn chat interface using AOT reasoning."""
    
    def __init__(self, model_name: str = "llama3"):
        self.reasoner = AOTReasoner(model_name)
        self.chat_history = []
        
    def add_to_history(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})
        
    def get_system_prompt(self) -> str:
        """Generate a system prompt incorporating chat history context."""
        if not self.chat_history:
            return "You are a helpful assistant using Atom of Thoughts reasoning."
        
        history_context = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in self.chat_history[-6:]  # Include last 6 messages for context
        ])
        
        return f"""You are a helpful assistant using Atom of Thoughts reasoning.
        
Here is the recent conversation history for context:

{history_context}

Use this context to inform your reasoning and responses."""
    
    def chat(self, user_input: str) -> str:
        """Process a user message and generate a response."""
        self.add_to_history("user", user_input)
        
        # Generate response using AOT reasoning with history context
        system_prompt = self.get_system_prompt()
        response = self.reasoner.reason(user_input)
        
        self.add_to_history("assistant", response)
        return response

class TerminalChat(cmd.Cmd):
    """Interactive terminal interface for AOT chat."""
    
    intro = "Welcome to AOT Chat! Type your questions or 'quit' to exit.\n"
    prompt = "You: "
    
    def __init__(self, model_name: str = "llama3"):
        super().__init__()
        self.chat = AOTChat(model_name)
        
    def default(self, line: str):
        """Handle user input."""
        if line.lower() in ('quit', 'exit', 'bye'):
            return self.do_quit(line)
        
        # Use AOT reasoning to generate response
        response = self.chat.chat(line)
        print(f"\nAOT: {response}\n")
        
    def do_quit(self, arg):
        """Exit the terminal chat."""
        print("Goodbye!")
        return True
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass

# FastAPI server for OpenAI-compatible API
app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Initialize AOT reasoner with the requested model
        reasoner = AOTReasoner(request.model)
        
        # Extract the latest user message
        user_messages = [msg for msg in request.messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        latest_user_message = user_messages[-1]["content"]
        
        # Generate system prompt from prior messages
        system_messages = [msg for msg in request.messages if msg["role"] == "system"]
        system_prompt = system_messages[-1]["content"] if system_messages else None
        
        # Use AOT reasoning to generate response
        response = reasoner.reason(latest_user_message)
        
        # Format response in OpenAI-compatible format
        return {
            "id": "chatcmpl-aot-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(latest_user_message) // 4,  # rough estimate
                "completion_tokens": len(response) // 4,  # rough estimate
                "total_tokens": (len(latest_user_message) + len(response)) // 4  # rough estimate
            }
        }
    except Exception as e:
        print(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="AOT Reasoning with Ollama")
    parser.add_argument("--model", type=str, default="llama3", 
                        help="Ollama model name to use (default: llama3)")
    parser.add_argument("--mode", type=str, choices=["terminal", "api"], default="terminal",
                        help="Run in terminal chat mode or API server mode")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for API server (default: 8000)")
    
    args = parser.parse_args()
    
    if args.mode == "terminal":
        # Start interactive terminal chat
        terminal = TerminalChat(args.model)
        terminal.cmdloop()
    else:
        # Start API server
        print(f"Starting OpenAI-compatible API server on port {args.port}")
        print(f"Using model: {args.model}")
        print("API endpoint: http://localhost:{args.port}/v1/chat/completions")
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    import time  # needed for the timestamp in API responses
    main()
