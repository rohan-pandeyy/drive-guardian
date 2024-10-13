//Import the THREE.js library
import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
// To allow for the camera to move around the scene
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
// To allow for importing the .gltf file
import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

//Create a Three.JS Scene
const scene = new THREE.Scene();
//create a new camera with positions and angles
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5; // Adjust camera position as needed

//Create a renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add OrbitControls for camera movement
const controls = new OrbitControls(camera, renderer.domElement);

// Add lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0); // Soft white ambient light
scene.add(ambientLight);

// Create 4 directional lights from 3 different diagonals
const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1);
directionalLight1.position.set(2, 2, 2); // Top-right diagonal
scene.add(directionalLight1);

const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1);
directionalLight2.position.set(-2, 2, 2); // Top-left diagonal
scene.add(directionalLight2);

const directionalLight3 = new THREE.DirectionalLight(0xffffff, 1);
directionalLight3.position.set(2, 2, -2); // Bottom-right diagonal
scene.add(directionalLight3);

const directionalLight4 = new THREE.DirectionalLight(0xffffff, 1);
directionalLight4.position.set(0, 5, 0); // Directly above
scene.add(directionalLight4);

// Load the GLTF model
const loader = new GLTFLoader();
loader.load(
  'drive-guardian.glb', // Replace with the actual path to your GLB file
  (gltf) => {
    const model = gltf.scene;
    scene.add(model);

    // Optional: Adjust model size or position
    // model.scale.set(0.5, 0.5, 0.5); // Example: Scale down the model
    // model.position.set(0, 1, 0); // Example: Move the model upwards

    // Render the scene
    animate();
  },
  (xhr) => {
    console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
  },
  (error) => {
    console.error('An error occurred while loading the GLTF model: ', error);
  }
);

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}