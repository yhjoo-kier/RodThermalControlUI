import asyncio
import websockets
import json
import numpy as np

async def test_ws():
    uri = "ws://localhost:8000/ws/simulation"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Start simulation
        await websocket.send(json.dumps({"command": "start"}))
        print("Sent start command")
        
        # Listen for 10 messages
        for i in range(10):
            response = await websocket.recv()
            data = json.loads(response)
            
            time = data['time']
            rl_temp = np.mean(data['rl']['temps'])
            mpc_temp = np.mean(data['mpc']['temps'])
            mpc_input = data['mpc']['input']
            
            print(f"Step {i}: Time={time:.2f}, RL_Mean={rl_temp:.2f}, MPC_Mean={mpc_temp:.2f}, MPC_Input={mpc_input:.2f}")
            
            if mpc_temp > 200 or mpc_temp < 0:
                print("FAIL: Temperature out of bounds!")
                return
                
        print("PASS: Data received and values are reasonable.")

if __name__ == "__main__":
    asyncio.run(test_ws())
