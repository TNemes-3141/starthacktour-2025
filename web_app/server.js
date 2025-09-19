const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS for frontend requests
app.use(cors());
app.use(express.json());

// Dummy object names and sample images
const objectNames = [
    'Alpha Sensor',
    'Beta Monitor', 
    'Gamma Tracker',
    'Delta Device',
    'Epsilon Scanner',
    'Zeta Detector',
    'Eta Probe',
    'Theta Beacon',
    'Iota Reader',
    'Kappa Collector'
];

// Sample placeholder images (using picsum for demo - replace with your actual image URLs)
const sampleImages = [
    'https://picsum.photos/300/200?random=1',
    'https://picsum.photos/300/200?random=2',
    'https://picsum.photos/300/200?random=3',
    'https://picsum.photos/300/200?random=4',
    'https://picsum.photos/300/200?random=5',
    'https://picsum.photos/300/200?random=6',
    'https://picsum.photos/300/200?random=7',
    'https://picsum.photos/300/200?random=8',
    'https://picsum.photos/300/200?random=9',
    'https://picsum.photos/300/200?random=10'
];

// Counter for generating unique IDs
let nextId = 1;
const serverStartTime = Date.now();

// Geneva area bounds for realistic coordinates
const GENEVA_BOUNDS = {
    lat: { min: 46.18, max: 46.22 },
    lng: { min: 6.10, max: 6.18 }
};

// Generate truly unique ID with timestamp and random component
function generateUniqueId() {
    const timestamp = Date.now();
    const random = Math.floor(Math.random() * 1000).toString().padStart(3, '0');
    const counter = String(nextId++).padStart(4, '0');
    
    // Format: OBJ-TIMESTAMP-COUNTER-RANDOM
    return `OBJ-${timestamp}-${counter}-${random}`;
}

// Alternative shorter unique ID generator (if you prefer shorter IDs)
function generateShortUniqueId() {
    const timeSinceStart = Date.now() - serverStartTime;
    const random = Math.floor(Math.random() * 100).toString().padStart(2, '0');
    const counter = String(nextId++).padStart(4, '0');
    
    // Format: OBJ-XXXX-XX (using base36 for compactness)
    return `OBJ-${timeSinceStart.toString(36).toUpperCase()}-${counter}-${random}`;
}

// Generate UUID-like unique ID (most robust)
function generateUUIDBasedId() {
    const chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let result = 'OBJ-';
    
    // Add 8 random characters
    for (let i = 0; i < 8; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    
    // Add timestamp component (last 6 digits of timestamp in base36)
    const timeComponent = (Date.now() % 1000000).toString(36).toUpperCase().padStart(4, '0');
    result += '-' + timeComponent;
    
    return result;
}

// Ensure ID is unique across all existing objects
function ensureUniqueId() {
    let id;
    let attempts = 0;
    const maxAttempts = 100;
    
    do {
        id = generateUUIDBasedId(); // Use the most robust generator
        attempts++;
        
        if (attempts > maxAttempts) {
            console.error('Failed to generate unique ID after', maxAttempts, 'attempts');
            // Fallback to timestamp-based ID
            id = `OBJ-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
            break;
        }
    } while (objects.some(obj => obj.id === id));
    
    return id;
}

// Store for our dummy objects
let objects = [];

// Generate random coordinate within Geneva bounds
function getRandomCoordinate() {
    const lat = GENEVA_BOUNDS.lat.min + Math.random() * (GENEVA_BOUNDS.lat.max - GENEVA_BOUNDS.lat.min);
    const lng = GENEVA_BOUNDS.lng.min + Math.random() * (GENEVA_BOUNDS.lng.max - GENEVA_BOUNDS.lng.min);
    return { lat: Number(lat.toFixed(6)), lng: Number(lng.toFixed(6)) };
}

// Generate random timestamp (some recent, some old)
function getRandomTimestamp() {
    const now = Date.now();
    const randomOffset = Math.random();
    
    if (randomOffset < 0.3) {
        // 30% chance: Very recent (0-2 minutes ago) - ACTIVE
        return new Date(now - Math.random() * 2 * 60 * 1000);
    } else if (randomOffset < 0.5) {
        // 20% chance: Recent but becoming inactive (3-7 minutes ago)
        return new Date(now - (3 + Math.random() * 4) * 60 * 1000);
    } else {
        // 50% chance: Old (10 minutes to 2 hours ago) - PAST
        return new Date(now - (10 + Math.random() * 110) * 60 * 1000);
    }
}

// Initialize dummy data
function initializeDummyData() {
    objects = [];
    
    // Create 8-12 random objects
    const numObjects = 8 + Math.floor(Math.random() * 5);
    
    console.log(`🏗️  Generating ${numObjects} objects with unique IDs...`);
    
    for (let i = 0; i < numObjects; i++) {
        const coord = getRandomCoordinate();
        const obj = {
            id: ensureUniqueId(),
            name: objectNames[i % objectNames.length],
            latitude: coord.lat,
            longitude: coord.lng,
            timestamp: getRandomTimestamp().toISOString(),
            image: sampleImages[i % sampleImages.length]
        };
        objects.push(obj);
        console.log(`  ✅ Created: ${obj.id} - ${obj.name}`);
    }
    
    // Verify uniqueness
    const allIds = objects.map(obj => obj.id);
    const uniqueIds = [...new Set(allIds)];
    const hasDuplicates = allIds.length !== uniqueIds.length;
    
    console.log(`🔍 ID Verification: ${objects.length} objects, ${uniqueIds.length} unique IDs`);
    if (hasDuplicates) {
        console.warn('⚠️  WARNING: Duplicate IDs detected!');
    } else {
        console.log('✅ All IDs are unique!');
    }
    
    console.log(`Generated ${objects.length} dummy objects`);
}

// Simulate object updates (some objects change position/timestamp)
function simulateUpdates() {
    objects.forEach(obj => {
        // 20% chance an object updates its position/timestamp
        if (Math.random() < 0.2) {
            const coord = getRandomCoordinate();
            obj.latitude = coord.lat;
            obj.longitude = coord.lng;
            obj.timestamp = new Date().toISOString(); // Update to current time
            console.log(`Updated ${obj.name} position and timestamp`);
        }
    });
    
    // 10% chance to add a completely new object
    if (Math.random() < 0.1 && objects.length < 15) {
        const coord = getRandomCoordinate();
        const newObj = {
            id: ensureUniqueId(),
            name: `New ${objectNames[Math.floor(Math.random() * objectNames.length)]}`,
            latitude: coord.lat,
            longitude: coord.lng,
            timestamp: new Date().toISOString(),
            image: sampleImages[Math.floor(Math.random() * sampleImages.length)]
        };
        objects.push(newObj);
        console.log(`Added new object: ${newObj.id} - ${newObj.name}`);
    }
}

// API Routes
app.get('/api/locations', (req, res) => {
    try {
        // Return all object data including ID
        res.json(objects);
    } catch (error) {
        console.error('Error serving locations:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        objectCount: objects.length,
        timestamp: new Date().toISOString()
    });
});

// Get server statistics
app.get('/api/stats', (req, res) => {
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);
    
    const activeObjects = objects.filter(obj => new Date(obj.timestamp) > fiveMinutesAgo);
    const pastObjects = objects.filter(obj => new Date(obj.timestamp) <= fiveMinutesAgo);
    
    // Check for unique IDs
    const allIds = objects.map(obj => obj.id);
    const uniqueIds = [...new Set(allIds)];
    const hasDuplicates = allIds.length !== uniqueIds.length;
    
    res.json({
        total: objects.length,
        active: activeObjects.length,
        past: pastObjects.length,
        uniqueIds: uniqueIds.length,
        hasDuplicateIds: hasDuplicates,
        duplicates: hasDuplicates ? allIds.filter((id, index) => allIds.indexOf(id) !== index) : [],
        lastUpdate: new Date().toISOString()
    });
});

// Check ID uniqueness endpoint
app.get('/api/check-ids', (req, res) => {
    const allIds = objects.map(obj => obj.id);
    const uniqueIds = [...new Set(allIds)];
    const duplicates = allIds.filter((id, index) => allIds.indexOf(id) !== index);
    
    res.json({
        totalObjects: objects.length,
        totalIds: allIds.length,
        uniqueIds: uniqueIds.length,
        hasDuplicates: allIds.length !== uniqueIds.length,
        duplicates: [...new Set(duplicates)], // Remove duplicate duplicates
        allIds: allIds,
        timestamp: new Date().toISOString()
    });
});

// Manual trigger for adding a new active object (for testing)
app.post('/api/add-object', (req, res) => {
    const coord = getRandomCoordinate();
    const newObj = {
        id: ensureUniqueId(),
        name: `Manual ${objectNames[Math.floor(Math.random() * objectNames.length)]}`,
        latitude: coord.lat,
        longitude: coord.lng,
        timestamp: new Date().toISOString(),
        image: sampleImages[Math.floor(Math.random() * sampleImages.length)]
    };
    
    objects.push(newObj);
    console.log(`Manually added: ${newObj.id} - ${newObj.name}`);
    
    res.json({ success: true, object: newObj });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Dummy backend server running on http://localhost:${PORT}`);
    console.log(`📍 API endpoint: http://localhost:${PORT}/api/locations`);
    console.log(`📊 Stats endpoint: http://localhost:${PORT}/api/stats`);
    console.log(`🔍 ID Check endpoint: http://localhost:${PORT}/api/check-ids`);
    console.log(`🏥 Health check: http://localhost:${PORT}/health`);
    console.log(`➕ Add object: POST http://localhost:${PORT}/api/add-object`);
    
    // Initialize dummy data
    initializeDummyData();
    
    // Simulate updates every 45 seconds
    setInterval(simulateUpdates, 45000);
    
    console.log('🔄 Object updates will occur every 45 seconds');
});