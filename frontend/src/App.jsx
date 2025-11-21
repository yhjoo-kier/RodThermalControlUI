import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import Rod3D from './components/Rod3D';
import Dashboard from './components/Dashboard';

function App() {
    const [data, setData] = useState(null);
    const [connected, setConnected] = useState(false);
    const [hCoeff, setHCoeff] = useState(15.0);
    const ws = useRef(null);

    useEffect(() => {
        ws.current = new WebSocket('ws://localhost:8000/ws/simulation');

        ws.current.onopen = () => {
            console.log('Connected to WebSocket');
            setConnected(true);
        };

        ws.current.onmessage = (event) => {
            const message = JSON.parse(event.data);

            setData(prev => {
                // Keep history of last 100 points
                const maxHistory = 100;
                const prevRlHistory = prev?.rl?.history || [];
                const prevMpcHistory = prev?.mpc?.history || [];

                const newRlHistory = [...prevRlHistory, { time: message.time, value: message.rl.reward }].slice(-maxHistory);
                const newMpcHistory = [...prevMpcHistory, { time: message.time, value: message.mpc.cost }].slice(-maxHistory);

                const prevRlInput = prev?.rl?.inputHistory || [];
                const prevMpcInput = prev?.mpc?.inputHistory || [];

                const newRlInput = [...prevRlInput, { time: message.time, value: message.rl.input }].slice(-maxHistory);
                const newMpcInput = [...prevMpcInput, { time: message.time, value: message.mpc.input }].slice(-maxHistory);

                return {
                    ...message,
                    rl: { ...message.rl, history: newRlHistory, inputHistory: newRlInput },
                    mpc: { ...message.mpc, history: newMpcHistory, inputHistory: newMpcInput }
                };
            });
        };

        ws.current.onclose = () => {
            console.log('Disconnected');
            setConnected(false);
        };

        return () => {
            if (ws.current) ws.current.close();
        };
    }, []);

    const handleStart = () => {
        if (ws.current) ws.current.send(JSON.stringify({ command: 'start' }));
    };

    const handleStop = () => {
        if (ws.current) ws.current.send(JSON.stringify({ command: 'stop' }));
    };

    const handleReset = () => {
        if (ws.current) ws.current.send(JSON.stringify({ command: 'reset' }));
    };

    const handleSpeed = (speed) => {
        if (ws.current) ws.current.send(JSON.stringify({ command: 'set_speed', speed }));
    };

    const handleHChange = (e) => {
        const val = parseFloat(e.target.value);
        setHCoeff(val);
        if (ws.current && !isNaN(val)) {
            ws.current.send(JSON.stringify({ command: 'set_params', h_convection: val }));
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 text-gray-800 font-sans">
            {/* Navbar */}
            <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">T</div>
                        <h1 className="text-xl font-bold text-gray-900 tracking-tight">
                            Thermal Control <span className="text-indigo-600">Simulator</span>
                        </h1>
                    </div>

                    <div className="flex items-center gap-4">
                        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${connected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></span>
                            {connected ? 'Connected' : 'Disconnected'}
                        </div>

                        <div className="h-6 w-px bg-gray-300 mx-2"></div>

                        <button onClick={handleStart} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md font-medium shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                            Start
                        </button>
                        <button onClick={handleStop} className="px-4 py-2 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-md font-medium shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                            Stop
                        </button>
                        <button onClick={handleReset} className="px-4 py-2 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-md font-medium shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                            Reset
                        </button>

                        <div className="h-6 w-px bg-gray-300 mx-2"></div>

                        <div className="flex items-center gap-2 bg-gray-100 rounded-md px-3 py-1">
                            <span className="text-xs font-medium text-gray-500">h:</span>
                            <input
                                type="number"
                                value={hCoeff}
                                onChange={handleHChange}
                                className="w-12 bg-transparent text-sm font-medium text-gray-700 focus:outline-none text-right"
                                step="1"
                            />
                            <span className="text-xs text-gray-400">W/m²K</span>
                        </div>

                        <div className="h-6 w-px bg-gray-300 mx-2"></div>

                        <div className="flex items-center bg-gray-100 rounded-md p-1">
                            {[1, 2, 5, 10].map((s) => (
                                <button
                                    key={s}
                                    onClick={() => handleSpeed(s)}
                                    className={`px-3 py-1 text-xs font-medium rounded-sm transition-all ${data?.speed === s ? 'bg-white shadow text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                                >
                                    {s}x
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* RL Section */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                                <h2 className="font-semibold text-gray-900 flex items-center gap-2">
                                    <span className="w-2 h-8 bg-blue-500 rounded-full"></span>
                                    RL Agent
                                </h2>
                                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Reinforcement Learning</span>
                            </div>

                            <div className="h-80 bg-gradient-to-b from-gray-100 to-white relative">
                                <Canvas camera={{ position: [0.2, 0.2, 0.35], fov: 45 }}>
                                    <ambientLight intensity={0.7} />
                                    <pointLight position={[10, 10, 10]} intensity={1.0} />
                                    <pointLight position={[-10, -10, -10]} intensity={0.5} />
                                    <Rod3D temperatures={data?.rl.temps || []} />
                                    <ContactShadows position={[0, -0.05, 0]} opacity={0.4} scale={10} blur={2} far={4.5} />
                                    <OrbitControls enableZoom={true} minDistance={0.2} maxDistance={1.0} />
                                    <Environment preset="city" />
                                </Canvas>

                                <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur px-3 py-2 rounded-lg shadow-sm border border-gray-200 text-xs text-gray-500">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="w-3 h-3 rounded-full bg-red-500"></span> High Temp (80°C+)
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="w-3 h-3 rounded-full bg-blue-500"></span> Low Temp (25°C)
                                    </div>
                                </div>
                            </div>
                        </div>

                        <Dashboard
                            data={data?.rl || { temps: [], input: 0, sensors: [] }}
                            label="RL Performance"
                            color="#3B82F6"
                        />
                    </div>

                    {/* MPC Section */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                                <h2 className="font-semibold text-gray-900 flex items-center gap-2">
                                    <span className="w-2 h-8 bg-emerald-500 rounded-full"></span>
                                    MPC Controller
                                </h2>
                                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Model Predictive Control</span>
                            </div>

                            <div className="h-80 bg-gradient-to-b from-gray-100 to-white relative">
                                <Canvas camera={{ position: [0.2, 0.2, 0.35], fov: 45 }}>
                                    <ambientLight intensity={0.7} />
                                    <pointLight position={[10, 10, 10]} intensity={1.0} />
                                    <pointLight position={[-10, -10, -10]} intensity={0.5} />
                                    <Rod3D temperatures={data?.mpc.temps || []} />
                                    <ContactShadows position={[0, -0.05, 0]} opacity={0.4} scale={10} blur={2} far={4.5} />
                                    <OrbitControls enableZoom={true} minDistance={0.2} maxDistance={1.0} />
                                    <Environment preset="city" />
                                </Canvas>
                            </div>
                        </div>

                        <Dashboard
                            data={data?.mpc || { temps: [], input: 0, sensors: [] }}
                            label="MPC Performance"
                            color="#10B981"
                        />
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
