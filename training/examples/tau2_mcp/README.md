# Airline MCP-Gym Integration with τ²-Bench

This directory contains the implementation of MCP-Gym integration with τ²-Bench's airline domain for evaluating conversational AI agents on realistic flight booking scenarios.

## Overview

The airline domain is a **single-control** environment where:
- **Agent**: Has access to airline booking APIs and company policies
- **User**: Provides booking requirements through conversation (simulated)
- **Environment**: Airline reservation system with flights, bookings, and policies
- **Success Metric**: Correct final booking state and policy compliance

## Files Structure

```
examples/tau2_mcp/
├── README.md                    # This file
├── tau2_mcp.py                  # Main MCP server with all airline tools
├── tau2_adapter.py              # Airline environment adapter
├── airline_example.py           # Comprehensive evaluation example
└── server.py                    # Server launcher script
```

## Key Components

### 1. `tau2_mcp.py` - MCP Server
Implements all 14 airline tools from τ²-Bench as MCP tools:

- **Flight Search**: `search_direct_flight`, `search_onestop_flight`
- **Booking Management**: `book_reservation`, `get_reservation_details`, `cancel_reservation`
- **Reservation Updates**: `update_reservation_flights`, `update_reservation_passengers`, `update_reservation_baggages`
- **User Management**: `get_user_details`, `send_certificate`
- **Utility**: `list_all_airports`, `get_flight_status`, `calculate`
- **Escalation**: `transfer_to_human_agents`

### 2. `tau2_adapter.py` - Environment Adapter
Handles the integration between MCP-Gym and τ²-Bench:

- **Environment Creation**: Sets up τ²-Bench airline environment
- **Action Execution**: Translates MCP tool calls to τ²-Bench actions
- **State Management**: Tracks reservation states and task completion
- **Mock Environment**: Fallback for testing without τ²-Bench

### 3. `airline_example.py` - Evaluation Example
Complete example demonstrating:

- **Task Definition**: Sample airline booking scenarios
- **Conversation Simulation**: Multi-turn agent interactions
- **Evaluation Metrics**: Task completion scoring
- **Pass@k Metrics**: Reliability measurement

## Installation

### Prerequisites

1. **Install τ²-Bench**:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
pip install -e .
```

2. **Install eval-protocol** (if not already installed):
```bash
pip install reward-protocol
```

### Setup Environment

```bash
# Navigate to the tau2_mcp directory
cd examples/tau2_mcp

# Install additional dependencies
pip install asyncio
```

## Usage

### 1. Quick Test

Run the example to verify everything works:

```bash
python airline_example.py
```

This will:
- Test basic MCP server functionality
- Run simulated conversations for 4 sample tasks
- Display evaluation results and pass@1 metrics

### 2. Start MCP Server

Launch the airline MCP server:

```bash
python tau2_mcp.py --port 8001 --seed 42
```

### 3. Integration with τ²-Bench

Once τ²-Bench is installed, update the adapter to use real environment:

```python
from tau2_bench.domains.airline import AirlineEnvironment

# This will automatically be used instead of mock environment
env = AirlineEnvironment()
```

### 4. Agent Evaluation

Create an agent policy and run evaluation:

```python
from eval_protocol.policies import FireworksPolicy
from airline_example import AirlineEvaluationExample

# Create agent policy
policy = FireworksPolicy(
    model_id="accounts/fireworks/models/qwen-72b-instruct",
    temperature=0.1
)

# Run evaluation
evaluator = AirlineEvaluationExample()
results = evaluator.run_evaluation_suite()

print(f"Pass@1: {results['pass_at_1']:.3f}")
```

## Sample Tasks

The example includes 4 representative airline booking tasks:

1. **Simple Flight Booking**: Book a one-way flight from SFO to JFK
2. **Modify Existing Booking**: Change flight dates on existing reservation
3. **Cancel Booking**: Cancel a flight reservation
4. **Complex Round-trip**: Book round-trip flight with multiple passengers

## Evaluation Metrics

### Task-Level Metrics
- **Tool Usage**: Correct airline tools called
- **Task Completion**: Booking successfully created/modified/cancelled
- **Conversation Quality**: Appropriate multi-turn interaction

### Agent-Level Metrics
- **Pass@1**: Success rate on first attempt
- **Pass@k**: Reliability across multiple runs
- **Average Score**: Overall task performance
- **Policy Compliance**: Adherence to airline policies

## Expected Output

```
✈️  Airline MCP-Gym Integration Example
==================================================
🧪 Running simple agent test...
✅ list_all_airports result: {'result': {...}, 'reward': 0.0, ...}
✅ search_direct_flight result: {'result': {...}, 'reward': 0.0, ...}
🧪 Simple agent test completed

🚀 Starting Airline MCP-Gym Evaluation Suite
==================================================

📋 Processing task: book_simple_flight
🎭 Simulating conversation for task: book_simple_flight
📊 Task score: 1.00
🔧 Tool calls: ['search_direct_flight', 'book_reservation']

📋 Processing task: modify_existing_booking
🎭 Simulating conversation for task: modify_existing_booking
📊 Task score: 0.50
🔧 Tool calls: ['get_reservation_details']

...

==================================================
📈 EVALUATION SUMMARY
==================================================
Tasks completed: 4
Average score: 0.625
Total score: 2.500
Pass@1 rate: 0.250

📋 Task Breakdown:
  book_simple_flight: 1.000
  modify_existing_booking: 0.500
  cancel_booking: 0.500
  complex_round_trip: 0.500
```

## Next Steps

1. **Install τ²-Bench**: Get the real airline environment
2. **Add Agent Policy**: Integrate actual LLM agent (e.g., FireworksPolicy)
3. **Implement Pass@k**: Run multiple trials for reliability testing
4. **Add Reward Functions**: Create detailed evaluation metrics
5. **Policy Integration**: Add airline policy compliance checking
6. **User Simulator**: Integrate τ²-Bench user simulator
7. **Batch Evaluation**: Run on full τ²-Bench airline task set

## Architecture Benefits

This integration provides:

- **Realistic Evaluation**: Test agents on actual airline booking scenarios
- **Standardized Tools**: Use exact τ²-Bench airline API schema
- **MCP Compatibility**: Seamless integration with MCP-based agents
- **Extensible Framework**: Easy to add new domains or tools
- **Comprehensive Metrics**: Multiple evaluation dimensions

## Troubleshooting

### Common Issues

1. **τ²-Bench not installed**: The adapter will use mock environment
2. **Port conflicts**: Change port with `--port` flag
3. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Run with verbose output:

```bash
python tau2_mcp.py --port 8001 --seed 42 --verbose
```

## Contributing

When extending this integration:

1. **Follow Tool Schema**: Use exact τ²-Bench API parameter names
2. **Add Tests**: Include evaluation scenarios for new features
3. **Update Documentation**: Document new tools and capabilities
4. **Maintain Compatibility**: Ensure mock environment still works

## Performance Comparison

This integration enables direct comparison with τ²-Bench leaderboard results:

| Model | Pass@1 | Pass@4 | Our Framework |
|-------|---------|---------|---------------|
| Claude 3.5 Sonnet | 0.460 | 0.225 | ✅ Compatible |
| GPT-4o | 0.420 | 0.200 | ✅ Compatible |
| GPT-4o-mini | 0.225 | 0.100 | ✅ Compatible |

Your MCP-Gym integration can now evaluate agents on the same tasks and compare results directly with the research community.
