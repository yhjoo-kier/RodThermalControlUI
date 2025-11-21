import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const Dashboard = ({ data, label, color }) => {
    const tempProfileData = data.temps.map((t, i) => ({
        x: i,
        temp: t
    }));

    const meanTemp = data.temps.length > 0
        ? (data.temps.reduce((a, b) => a + b, 0) / data.temps.length).toFixed(1)
        : "0.0";

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex justify-between items-end mb-6">
                <div>
                    <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">Metrics</h3>
                    <p className="text-2xl font-bold text-gray-900">{label}</p>
                </div>
                <div className="flex gap-4">
                    <div className="text-right">
                        <p className="text-xs text-gray-400 mb-1">Input Power</p>
                        <p className="text-xl font-mono font-semibold text-gray-900">
                            {data.input.toFixed(1)} <span className="text-sm text-gray-500 font-sans">W</span>
                        </p>
                    </div>
                    <div className="text-right pl-4 border-l border-gray-100">
                        <p className="text-xs text-gray-400 mb-1">Mean Temp</p>
                        <p className="text-xl font-mono font-semibold text-gray-900">
                            {meanTemp} <span className="text-sm text-gray-500 font-sans">°C</span>
                        </p>
                    </div>
                </div>
            </div>

            <div className="mb-2">
                <h4 className="text-xs font-medium text-gray-400 mb-3">Temperature Profile (°C)</h4>
                <div className="h-32 w-full mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={tempProfileData}>
                            <defs>
                                <linearGradient id={`color${label}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={color} stopOpacity={0.1} />
                                    <stop offset="95%" stopColor={color} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
                            <XAxis dataKey="x" hide />
                            <YAxis domain={[20, 100]} hide />
                            <Tooltip contentStyle={{ fontSize: '12px' }} />
                            <Area type="monotone" dataKey="temp" stroke={color} fillOpacity={1} fill={`url(#color${label})`} strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <h4 className="text-xs font-medium text-gray-400 mb-3">Control Objective (Cost/Reward)</h4>
                <div className="h-32 w-full mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data.history || []}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
                            <XAxis dataKey="time" hide />
                            <YAxis hide />
                            <Tooltip contentStyle={{ fontSize: '12px' }} />
                            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <h4 className="text-xs font-medium text-gray-400 mb-3">Input Power (W)</h4>
                <div className="h-32 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data.inputHistory || []}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
                            <XAxis dataKey="time" hide />
                            <YAxis domain={[0, 50]} hide />
                            <Tooltip contentStyle={{ fontSize: '12px' }} />
                            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
