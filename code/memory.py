import numpy as np
import json

# Global target value for optimization (set to 0 as default)


# ============================================================
# === Memory with LLM-Generated Natural Language Summaries
# ============================================================
class OptimizationMemory:
    """
    Memory system based on Evo-Memory paper principles:
    - LLM-generated natural language summaries
    - 4 significant figures max for readability
    - Experience reuse over conversational recall
    """

    def __init__(self, system_prompt, max_recent_turns=3, replay_buffer_size=5,
                 max_population=5, milestone_frequency=20):
        self.system_prompt = system_prompt
        self.max_recent_turns = max_recent_turns
        self.replay_buffer_size = replay_buffer_size
        self.max_population = max_population
        self.milestone_frequency = milestone_frequency

        # Global tracking
        self.best_score = float('inf')
        self.best_params = None
        
        # Enhanced: Convergence curve with parameters & reasoning
        self.convergence_curve = []

        # Compressed memory
        self.best_attempts = []

        # NEW: Track all evaluated parameter sets for genetic selection
        self.population = []

        # Agent-specific memory
        self.agent_memories = {}

        # Milestones with parameters
        self.milestones = []
        self.last_milestone_iter = 0

        # Debug tracking
        self.total_iterations = 0

        # NEW: Hypothesis tracking with scores
        self.hypotheses = []
        self._hypothesis_counter = 0

    def initialize_agent_memory(self, agent_id, character):
        """Initialize memory for a specific agent"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = {
                'character': character,
                'recent_proposals': [],
                'best_personal_score': float('inf'),
                'best_personal_params': None,
                'successful_strategies': [],
            }

    def add_to_replay_buffer(self, iteration, proposal, score, params):
        """Add experience to GLOBAL replay buffer with full context"""
        experience = {
            'iteration': iteration,
            'score': score,
            'reasoning': proposal.get('reasoning', 'No reasoning')[:2000],
            'params': params.copy(),
            'ape': 100 * abs(self.best_score - score) / (abs(self.best_score) + 1e-10) if abs(self.best_score) < float('inf') else float('inf'),
            'improvement_from_best': score - self.best_score
        }

        self.best_attempts.append(experience)
        self.best_attempts.sort(key=lambda x: x['score'])
        
        if len(self.best_attempts) > self.replay_buffer_size:
            self.best_attempts = self.best_attempts[:self.replay_buffer_size]

    def add_hypothesis(self, iteration, proposal, score, params):
        """Store hypothesis with its score for future reference"""
        hypothesis_data = proposal.get('hypothesis', {})
        if not hypothesis_data:
            return
        
        self._hypothesis_counter += 1
        hypothesis_entry = {
            'hypothesis_id': self._hypothesis_counter,
            'iteration': iteration,
            'score': score,
            'scientific_rationale': hypothesis_data.get('scientific_rationale', '')[:500],
            'test_method': hypothesis_data.get('test_method', '')[:500],
            'params': params.copy(),
            'reasoning': proposal.get('reasoning', '')[:500]
        }
        self.hypotheses.append(hypothesis_entry)
        
        if len(self.hypotheses) > 10:
            self.hypotheses.sort(key=lambda x: x['score'])
            self.hypotheses = self.hypotheses[:10]

    def get_top_hypotheses(self, top_n=3):
        """Get top N hypotheses by score"""
        if not self.hypotheses:
            return []
        sorted_hyps = sorted(self.hypotheses, key=lambda x: x['score'])
        return sorted_hyps[:top_n]

    def get_hypothesis_context(self):
        """Generate context string about top hypotheses for prompts"""
        top_hyps = self.get_top_hypotheses(3)
        if not top_hyps:
            return ""
        
        lines = ["**📝 Top Hypotheses (ranked by score):**"]
        for i, h in enumerate(top_hyps, 1):
            lines.append(f"{i}. Hypothesis #{h['hypothesis_id']} (score={self._fmt(h['score'])})")
            lines.append(f"   Rationale: {h['scientific_rationale'][:200]}...")
            lines.append(f"   Test method: {h['test_method'][:200]}...")
        return "\n".join(lines)

    def _fmt(self, value):
        """Format number to 4 significant figures"""
        if value is None:
            return "N/A"
        if abs(value) < 0.0001:
            return "0"
        return f"{value:.4g}"

    def _params_to_text(self, params):
        """Convert params to compact text representation (4 sig figs)"""
        if params is None:
            return "N/A"
        if isinstance(params, dict):
            return ", ".join([f"{k}={self._fmt(v)}" for k, v in params.items()])
        elif isinstance(params, np.ndarray):
            return f"[{', '.join([self._fmt(p) for p in params])}]"
        return str(params)

    def _generate_llm_summary(self, prompt, call_llm_func, parameters):
        """
        Use LLM to generate natural language summary
        
        Args:
            prompt: The prompt to send to LLM
            call_llm_func: Function to call LLM (from call_llm.py)
            parameters: Parameter config needed for LLM call
        """
        try:
            # Create simple message for LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise, natural language summaries. Keep responses under 100 words."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call LLM (it will return a dict with 'reasoning' and 'action')
            # We only want the text summary, so we'll extract from response
            response = call_llm_func(messages, parameters)
            
            # Extract summary from reasoning or action
            if isinstance(response, dict):
                summary = response.get('reasoning', str(response))
            else:
                summary = str(response)
            
            # Clean up and truncate
            summary = summary.strip()
            return summary
            
        except Exception as e:
            print(f"  ⚠️  LLM summary generation failed: {e}")
            return "Summary generation failed"

    def update_agent_memory(self, agent_id, iteration, proposal, score, params):
        """Update agent-specific memory with parameters"""
        if agent_id not in self.agent_memories:
            return
        
        agent_mem = self.agent_memories[agent_id]
        
        agent_mem['recent_proposals'].append({
            'iteration': iteration,
            'score': score,
            'reasoning': proposal.get('reasoning', ''),
            'params': params.copy()
        })
        
        if len(agent_mem['recent_proposals']) > 2:
            agent_mem['recent_proposals'].pop(0)
        
        # Update personal best
        if score < agent_mem['best_personal_score']:
            agent_mem['best_personal_score'] = score
            agent_mem['best_personal_params'] = params.copy()
            
            agent_mem['successful_strategies'].append({
                'score': score,
            'reasoning': proposal.get('reasoning', ''),
                'params': params.copy()
            })
            if len(agent_mem['successful_strategies']) > 2:
                agent_mem['successful_strategies'].pop(0)

    def create_milestone(self, iteration, generation, call_llm_func=None, parameters=None):
        """
        Create milestone with LLM-generated natural language summary
        
        Args:
            iteration: Current iteration
            generation: Current generation
            call_llm_func: Function to call LLM for summary generation
            parameters: Parameter config for LLM
        """
        if len(self.convergence_curve) == 0:
            print(f"  ⚠️  Cannot create milestone at iter {iteration}: convergence_curve is empty")
            return

        recent_window = min(10, len(self.convergence_curve))
        recent_entries = self.convergence_curve[-recent_window:]
        recent_scores = [e['score'] for e in recent_entries]

        # Calculate statistics
        avg_recent = float(np.mean(recent_scores))
        improvement_rate = float((recent_scores[0] - recent_scores[-1]) / recent_window) if len(recent_scores) > 1 else 0
        
        # Get top recent strategy
        top_strategy = self._get_top_recent_strategy(recent_entries)
        
        # Generate LLM summary if function provided
        natural_summary = None
        if call_llm_func is not None and parameters is not None:
            summary_prompt = self._create_summary_prompt(
                generation, avg_recent, improvement_rate, top_strategy
            )
            natural_summary = self._generate_llm_summary(summary_prompt, call_llm_func, parameters)
        else:
            # Fallback to simple summary
            natural_summary = (
                f"Generation {generation}: best score {self._fmt(self.best_score)}, "
                f"avg recent {self._fmt(avg_recent)}, improvement rate {self._fmt(improvement_rate)}/iter, "
                f"best params: {self._params_to_text(self.best_params)}"
            )

        milestone = {
            'iteration': iteration,
            'generation': generation,
            'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'avg_recent': avg_recent,
            'improvement_rate': improvement_rate,
            'top_recent_strategy': top_strategy,
            'natural_summary': natural_summary  # ⭐ LLM-generated summary
        }

        self.milestones.append(milestone)
        
        if len(self.milestones) > 3:
            self.milestones = self.milestones[-3:]
        
        self.last_milestone_iter = iteration
        
        print(f"  ✅ Milestone at iter {iteration}:")
        print(f"     {natural_summary}")

    def _create_summary_prompt(self, generation, avg_recent, improvement_rate, top_strategy):
        """Create prompt for LLM to generate milestone summary"""
        prompt = f"""Summarize this optimization milestone in 1-2 sentences (max 100 words):

Generation: {generation}
        Best score achieved: {self._fmt(self.best_score)} (target: 0)
Average recent score: {self._fmt(avg_recent)}
Improvement rate: {self._fmt(improvement_rate)} per iteration
Best parameters: {self._params_to_text(self.best_params)}

Top performing strategy: {top_strategy['reasoning'] if top_strategy else 'N/A'}

Focus on: what was achieved, the trend, and what approach worked best. Be concise and informative."""
        return prompt

    def _get_top_recent_strategy(self, recent_entries):
        """Extract the best-performing strategy from recent entries"""
        if not recent_entries:
            return None
        
        best = min(recent_entries, key=lambda x: x['score'])
        return {
            'score': best['score'],
            'reasoning': best['reasoning'][:2000],
            'params': best['params']
        }

    def get_compressed_global_summary(self, call_llm_func=None, parameters=None):
        """
        Global summary with optional LLM-generated insights
        
        If call_llm_func provided, generates natural language summary
        Otherwise returns formatted statistics
        """
        if not self.best_attempts:
            return ""

        # If LLM available, generate natural summary
        if call_llm_func is not None and parameters is not None:
            summary_data = {
                'top_3_scores': [self._fmt(e['score']) for e in self.best_attempts[:3]],
                'top_3_apes': [self._fmt(e['ape']) for e in self.best_attempts[:3]],
                'top_3_strategies': [e['reasoning'][:2000] for e in self.best_attempts[:3]],
                'best_params': self._params_to_text(self.best_attempts[0]['params'])
            }
            
            prompt = f"""Summarize the top 3 optimization attempts in 2-3 sentences (max 100 words):

1. Score: {summary_data['top_3_scores'][0]}, Error: {summary_data['top_3_apes'][0]}%
   Strategy: {summary_data['top_3_strategies'][0]}
   
2. Score: {summary_data['top_3_scores'][1] if len(self.best_attempts) > 1 else 'N/A'}
   Strategy: {summary_data['top_3_strategies'][1] if len(self.best_attempts) > 1 else 'N/A'}
   
3. Score: {summary_data['top_3_scores'][2] if len(self.best_attempts) > 2 else 'N/A'}
   Strategy: {summary_data['top_3_strategies'][2] if len(self.best_attempts) > 2 else 'N/A'}

Best parameters: {summary_data['best_params']}

Focus on: what patterns emerge, which approaches work best, and key insights."""
            
            llm_summary = self._generate_llm_summary(prompt, call_llm_func, parameters)
            return f"📊 TOP PERFORMERS:\n{llm_summary}"
        
        # Fallback: Simple formatted summary
        lines = ["📊 TOP PERFORMERS:"]
        for i, exp in enumerate(self.best_attempts[:3], 1):
            lines.append(
                f"{i}. f={self._fmt(exp['score'])} (error {self._fmt(exp['ape'])}%) | "
                f"Strategy: {exp['reasoning'][:2000]} | "
                f"Params: {self._params_to_text(exp['params'])}"
            )
        return "\n".join(lines)

    def get_compressed_agent_summary(self, agent_id):
        """Agent-specific summary (simple format)"""
        if agent_id not in self.agent_memories:
            return ""
        
        agent_mem = self.agent_memories[agent_id]
        lines = []
        
        # Personal best
        if agent_mem['best_personal_score'] < float('inf'):
            lines.append(f"🎯 Your best: {self._fmt(agent_mem['best_personal_score'])}")
        
        # Last successful strategy
        if agent_mem['successful_strategies']:
            strat = agent_mem['successful_strategies'][-1]
            lines.append(f"✓ Last success: {strat['reasoning'][:2000]}")
        
        return "\n".join(lines)

    def get_compressed_milestone_summary(self):
        """Natural language milestone summary (from LLM)"""
        if not self.milestones:
            return ""
        
        latest = self.milestones[-1]
        return f"📈 Latest Milestone:\n{latest['natural_summary']}"

    def get_context_message(self, iteration, agent_id=None, generation=None, epoch=None, 
                           call_llm_func=None, parameters=None):
        """
        Generate context message with optional LLM-enhanced summaries
        
        Args:
            iteration: Current iteration
            agent_id: Agent identifier
            generation: Current generation
            epoch: Current epoch
            call_llm_func: Optional LLM function for enhanced summaries
            parameters: Parameter config for LLM
        """
        if iteration == 0:
            return "🚀 Starting optimization. Explore the search space to find promising regions."

        context_parts = []
        
        # Core status
        gen_str = f"Generation {(generation+1) if generation is not None else 'N/A'}"
        epoch_str = f"Epoch {(epoch+1) if epoch is not None else 'N/A'}"
        context_parts.append(
            f"📍 Iteration {iteration} | {gen_str} | {epoch_str}\n"
            f"Current best: f={self._fmt(self.best_score)} (target: 0)"
        )
        
        # Global summary (with optional LLM enhancement)
        global_summary = self.get_compressed_global_summary(call_llm_func, parameters)
        if global_summary:
            context_parts.append(f"\n{global_summary}")
        
        # Agent-specific
        if agent_id is not None:
            agent_summary = self.get_compressed_agent_summary(agent_id)
            if agent_summary:
                context_parts.append(f"\n{agent_summary}")
        
        # Milestone (already LLM-generated)
        milestone_summary = self.get_compressed_milestone_summary()
        if milestone_summary:
            context_parts.append(f"\n{milestone_summary}")

        # Hypothesis context (ranked by score)
        hypothesis_context = self.get_hypothesis_context()
        if hypothesis_context:
            context_parts.append(f"\n{hypothesis_context}")

        return "\n".join(context_parts)

    def update(self, iteration, proposal, score, params, agent_id=None, generation=None,
               call_llm_func=None, parameters=None):
        """
        Update all memory systems with FULL context
        
        Args:
            iteration: Current iteration
            proposal: Proposal dict with reasoning
            score: Achieved score
            params: Parameters used
            agent_id: Agent identifier
            generation: Current generation
            call_llm_func: Optional LLM function for milestone summaries
            parameters: Parameter config for LLM
        """
        # Store full convergence entry
        convergence_entry = {
            'iteration': iteration,
            'score': score,
            'params': params.copy(),
            'reasoning': proposal.get('reasoning', '')[:2000]
        }
        self.convergence_curve.append(convergence_entry)
        
        # Keep last 20 only
        if len(self.convergence_curve) > 20:
            self.convergence_curve = self.convergence_curve[-20:]

        # Update global best
        if score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()

        # Add to global replay buffer
        self.add_to_replay_buffer(iteration, proposal, score, params)

        # Add to population for GA
        self.population.append({
            'params': params.copy(),
            'score': score,
            'iteration': iteration,
            'reasoning': proposal.get('reasoning', '')[:2000]
        })

        # KEEP ONLY TOP N (minimal!)
        self.population.sort(key=lambda x: x['score'])
        if len(self.population) > self.max_population:
            self.population = self.population[:self.max_population]

        # Update agent-specific memory
        if agent_id is not None:
            self.update_agent_memory(agent_id, iteration, proposal, score, params)

        # Add hypothesis to tracking (if present in proposal)
        self.add_hypothesis(iteration, proposal, score, params)

        # Create milestone if needed (with LLM summary)
        if iteration > 0 and iteration % self.milestone_frequency == 0:
            self.create_milestone(iteration, generation if generation is not None else 0,
                                call_llm_func, parameters)

    def get_history(self, iteration, agent_id=None, generation=None, epoch=None,
                   call_llm_func=None, parameters=None):
        """Get conversation history with optional LLM-enhanced context"""
        return [
            {
                "role": "user", 
                "content": self.get_context_message(iteration, agent_id, generation, epoch,
                                                   call_llm_func, parameters)
            }
        ]

    def get_best_parents(self, n=2):
        """Get top N parents for genetic crossover"""
        return self.population[:n] if len(self.population) >= n else []

    def get_population_diversity(self):
        """Calculate parameter diversity in population (for diagnostics)"""
        if len(self.population) < 2:
            return 0.0

        # Calculate average pairwise distance
        diversities = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                p1 = self.population[i]['params']
                p2 = self.population[j]['params']
                # Euclidean distance in parameter space
                diff = sum((p1[k] - p2[k])**2 for k in p1.keys())
                diversities.append(np.sqrt(diff))

        return np.mean(diversities) if diversities else 0.0

    def get_memory_stats(self):
        """Get statistics about memory usage"""
        return {
            'best_attempts_count': len(self.best_attempts),
            'agent_memories_count': len(self.agent_memories),
            'milestones_count': len(self.milestones),
            'convergence_length': len(self.convergence_curve),
            'population_size': len(self.population),
            'population_diversity': self.get_population_diversity(),
            'best_score': self.best_score,
            'best_params': self._params_to_text(self.best_params)
        }

    def clear_old_data(self):
        """Aggressive cleanup to prevent memory bloat"""
        if len(self.convergence_curve) > 20:
            self.convergence_curve = self.convergence_curve[-20:]
        
        if len(self.best_attempts) > self.replay_buffer_size:
            self.best_attempts = self.best_attempts[:self.replay_buffer_size]
        
        if len(self.milestones) > 3:
            self.milestones = self.milestones[-3:]
        
        for agent_id, mem in self.agent_memories.items():
            if len(mem['recent_proposals']) > 2:
                mem['recent_proposals'] = mem['recent_proposals'][-2:]
            if len(mem['successful_strategies']) > 2:
                mem['successful_strategies'] = mem['successful_strategies'][-2:]

    def save_to_json(self, filename="memory_state.json"):
        """Save the current memory state to a JSON file"""
        import os

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        memory_state = {
            'best_score': self.best_score,
            'best_params': convert_numpy(self.best_params),
            'convergence_curve': convert_numpy(self.convergence_curve[-10:]),
            'best_attempts': convert_numpy(self.best_attempts[:3]),
            'milestones': convert_numpy(self.milestones[-3:]),
            'population': convert_numpy(self.population),
            'hypotheses': convert_numpy(self.hypotheses[:10]),
            'memory_stats': self.get_memory_stats()
        }

        try:
            os.makedirs("memories", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(memory_state, f, indent=4)
        except Exception as e:
            print(f"Error saving memory to {filename}: {str(e)}")

    def save_iteration_memory(self, iteration):
        """Save memory state for a specific iteration"""
        filename = f"memories/iteration_{iteration:04d}.json"
        self.save_to_json(filename)

    @staticmethod
    def load_from_json(filename):
        """Load memory state from JSON file and restore"""
        import os
        
        if not os.path.exists(filename):
            print(f"Warning: Memory file {filename} not found")
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create new instance (system_prompt will be set by caller)
        memory = OptimizationMemory(system_prompt="")
        
        # Restore all state
        memory.best_score = data.get('best_score', float('inf'))
        memory.best_params = data.get('best_params', {})
        memory.convergence_curve = data.get('convergence_curve', [])
        memory.best_attempts = data.get('best_attempts', [])
        memory.population = data.get('population', [])
        memory.hypotheses = data.get('hypotheses', [])
        memory.milestones = data.get('milestones', [])
        
        # Restore hypothesis counter
        if memory.hypotheses:
            memory._hypothesis_counter = max(h.get('hypothesis_id', 0) for h in memory.hypotheses)
        
        print(f"  ✓ Loaded memory: best_score={memory.best_score:.4f}, pop_size={len(memory.population)}, hyps={len(memory.hypotheses)}")
        
        return memory


# ============================================================
# === Context Size Estimator (for debugging)
# ============================================================
def estimate_token_count(text):
    """Rough estimate of token count (1 token ≈ 4 chars)"""
    return len(text) // 4

def check_context_size(memory, iteration, agent_id, generation, epoch):
    """Debug function to check if context is within limits"""
    history = memory.get_history(iteration, agent_id, generation, epoch)
    
    total_chars = 0
    for msg in history:
        total_chars += len(msg['content'])
    
    estimated_tokens = estimate_token_count(total_chars)
    
    print(f"  📏 Context size: ~{estimated_tokens} tokens ({total_chars} chars)")
    
    if estimated_tokens > 2000:
        print(f"  ⚠️  WARNING: Context too large! Risk of overflow.")
    elif estimated_tokens > 1000:
        print(f"  ⚠️  Context getting large. Consider compression.")
    else:
        print(f"  ✅ Context size OK")
    
    return estimated_tokens