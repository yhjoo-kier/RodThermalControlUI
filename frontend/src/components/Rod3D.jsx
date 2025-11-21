import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const Rod3D = ({ temperatures, length = 0.2, diameter = 0.01 }) => {
    const meshRef = useRef();

    // Create geometry: Cylinder
    const n_segments = temperatures.length > 0 ? temperatures.length : 10;

    const geometry = useMemo(() => {
        // CylinderGeometry(radiusTop, radiusBottom, height, radialSegments, heightSegments)
        // Increased diameter for better visibility (diameter * 1.5)
        const geo = new THREE.CylinderGeometry(diameter * 1.5, diameter * 1.5, length, 32, n_segments - 1, true);
        geo.rotateZ(-Math.PI / 2);
        return geo;
    }, [length, diameter, n_segments]);

    // Update colors based on temperature
    useFrame(() => {
        if (meshRef.current && temperatures.length > 0) {
            const minTemp = 25;
            const maxTemp = 80; // Reduced max temp range to make colors more distinct for typical operation

            // Helper to get color from value
            const getColor = (t) => {
                // Jet-like map: Blue -> Green -> Red
                const clampedT = Math.max(minTemp, Math.min(maxTemp, t));
                const norm = (clampedT - minTemp) / (maxTemp - minTemp);

                const color = new THREE.Color();
                // HSL: Blue (0.66) to Red (0.0)
                color.setHSL((1.0 - norm) * 0.66, 1.0, 0.5);
                return color;
            };

            const count = meshRef.current.geometry.attributes.position.count;
            const colorAttr = new Float32Array(count * 3);
            const positions = meshRef.current.geometry.attributes.position.array;

            for (let i = 0; i < count; i++) {
                const x = positions[i * 3]; // Local x (since we rotated)
                // x ranges from -length/2 to length/2
                const normalizedPos = (x + length / 2) / length; // 0 to 1

                // Map position to temperature index
                // Clamp index to be safe
                const index = Math.min(n_segments - 1, Math.max(0, Math.floor(normalizedPos * n_segments)));

                const temp = temperatures[index] !== undefined ? temperatures[index] : 25;
                const c = getColor(temp);

                colorAttr[i * 3] = c.r;
                colorAttr[i * 3 + 1] = c.g;
                colorAttr[i * 3 + 2] = c.b;
            }

            meshRef.current.geometry.setAttribute('color', new THREE.BufferAttribute(colorAttr, 3));
            meshRef.current.geometry.attributes.color.needsUpdate = true;
        }
    });

    return (
        <mesh ref={meshRef} geometry={geometry}>
            <meshStandardMaterial vertexColors roughness={0.5} metalness={0.8} />
        </mesh>
    );
};

export default Rod3D;
