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

    // Training states
    const [showTrainingModal, setShowTrainingModal] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [trainingConfig, setTrainingConfig] = useState({
        total_timesteps: 100000,
        n_envs: 4,
        checkpoint_freq: 10000
    });
    const trainingWs = useRef(null);

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

    // Training WebSocket
    useEffect(() => {
        trainingWs.current = new WebSocket('ws://localhost:8000/ws/training');

        trainingWs.current.onopen = () => {
            console.log('Connected to training WebSocket');
        };

        trainingWs.current.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'training_update') {
                setTrainingStatus(message.status);

                // Close modal if training completed or errored
                if (message.completed || message.error) {
                    setTimeout(() => {
                        setShowTrainingModal(false);
                    }, 3000);
                }
            }
        };

        trainingWs.current.onclose = () => {
            console.log('Training WebSocket disconnected');
        };

        return () => {
            if (trainingWs.current) trainingWs.current.close();
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

    // Training handlers
    const handleStartTraining = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainingConfig)
            });
            const result = await response.json();
            console.log('Training started:', result);
            setShowTrainingModal(true);
        } catch (error) {
            console.error('Failed to start training:', error);
            alert('Failed to start training: ' + error.message);
        }
    };

    const handleStopTraining = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/train/stop', {
                method: 'POST'
            });
            const result = await response.json();
            console.log('Training stopped:', result);
        } catch (error) {
            console.error('Failed to stop training:', error);
        }
    };

    const formatTime = (seconds) => {
        if (!seconds || seconds < 0) return '0s';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
        if (mins > 0) return `${mins}m ${secs}s`;
        return `${secs}s`;
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

                        <button
                            onClick={handleStartTraining}
                            disabled={trainingStatus?.is_training}
                            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-400 text-white rounded-md font-medium shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:cursor-not-allowed"
                        >
                            {trainingStatus?.is_training ? '🔄 Training...' : '🎓 Train RL Model'}
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

            {/* Training Modal */}
            {showTrainingModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        {/* Modal Header */}
                        <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-600 to-indigo-600">
                            <div className="flex justify-between items-center">
                                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                                    <span>🎓</span> RL Model Training
                                </h2>
                                <button
                                    onClick={() => setShowTrainingModal(false)}
                                    className="text-white hover:text-gray-200 transition-colors"
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>

                        {/* Modal Body */}
                        <div className="p-6 space-y-6">
                            {/* Configuration */}
                            <div className="bg-gray-50 rounded-lg p-4">
                                <h3 className="text-sm font-semibold text-gray-700 mb-3">Training Configuration</h3>
                                <div className="grid grid-cols-3 gap-4">
                                    <div>
                                        <label className="text-xs text-gray-500">Total Timesteps</label>
                                        <input
                                            type="number"
                                            value={trainingConfig.total_timesteps}
                                            onChange={(e) => setTrainingConfig({...trainingConfig, total_timesteps: parseInt(e.target.value)})}
                                            disabled={trainingStatus?.is_training}
                                            className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm disabled:bg-gray-100"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs text-gray-500">Parallel Envs</label>
                                        <input
                                            type="number"
                                            value={trainingConfig.n_envs}
                                            onChange={(e) => setTrainingConfig({...trainingConfig, n_envs: parseInt(e.target.value)})}
                                            disabled={trainingStatus?.is_training}
                                            className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm disabled:bg-gray-100"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs text-gray-500">Checkpoint Freq</label>
                                        <input
                                            type="number"
                                            value={trainingConfig.checkpoint_freq}
                                            onChange={(e) => setTrainingConfig({...trainingConfig, checkpoint_freq: parseInt(e.target.value)})}
                                            disabled={trainingStatus?.is_training}
                                            className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm disabled:bg-gray-100"
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Progress */}
                            {trainingStatus && (
                                <div className="space-y-4">
                                    {/* Progress Bar */}
                                    <div>
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-sm font-medium text-gray-700">Progress</span>
                                            <span className="text-sm font-bold text-indigo-600">
                                                {trainingStatus.progress.toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                                            <div
                                                className="bg-gradient-to-r from-purple-600 to-indigo-600 h-3 rounded-full transition-all duration-300"
                                                style={{ width: `${trainingStatus.progress}%` }}
                                            />
                                        </div>
                                        <div className="flex justify-between items-center mt-1 text-xs text-gray-500">
                                            <span>{trainingStatus.current_step.toLocaleString()} / {trainingStatus.total_steps.toLocaleString()} steps</span>
                                        </div>
                                    </div>

                                    {/* Stats Grid */}
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-blue-50 rounded-lg p-4">
                                            <div className="text-xs text-blue-600 font-medium mb-1">Elapsed Time</div>
                                            <div className="text-2xl font-bold text-blue-700">
                                                {formatTime(trainingStatus.elapsed_time)}
                                            </div>
                                        </div>
                                        <div className="bg-purple-50 rounded-lg p-4">
                                            <div className="text-xs text-purple-600 font-medium mb-1">Estimated Remaining</div>
                                            <div className="text-2xl font-bold text-purple-700">
                                                {formatTime(trainingStatus.estimated_remaining)}
                                            </div>
                                        </div>
                                        {trainingStatus.current_reward !== null && (
                                            <div className="bg-green-50 rounded-lg p-4 col-span-2">
                                                <div className="text-xs text-green-600 font-medium mb-1">Current Reward</div>
                                                <div className="text-2xl font-bold text-green-700">
                                                    {trainingStatus.current_reward.toFixed(2)}
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Status Message */}
                                    <div className="bg-gray-100 rounded-lg p-4">
                                        <div className="text-xs text-gray-500 font-medium mb-1">Status</div>
                                        <div className="text-sm text-gray-700 font-mono">
                                            {trainingStatus.message}
                                        </div>
                                    </div>

                                    {/* Stop Button */}
                                    {trainingStatus.is_training && (
                                        <button
                                            onClick={handleStopTraining}
                                            className="w-full px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium shadow-sm transition-colors"
                                        >
                                            Stop Training
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
