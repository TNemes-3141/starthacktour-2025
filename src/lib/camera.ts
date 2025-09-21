// lib/camera.ts

export type LatLng = { lat: number; lng: number };

export const FOV = {
    // 26 mm FF-equivalent → angles on 36x24 mm frame
    horizontalDeg: 69.3903,
    verticalDeg: 49.5503,
    diagonalDeg: 79.5243,

    // Half-angles (often what you need for cone math)
    half: {
        horizontalDeg: 34.6952,
        verticalDeg: 24.7751,
        diagonalDeg: 39.7622,
    },
};

export const CAMERA = {
    position: { lat: 47.30658844506907, lng: 9.431777965149525 } as LatLng,

    /**
     * Direction the camera is pointing, in degrees azimuth:
     * 0 = North, 90 = East, 180 = South, 270 = West
     */
    azimuthDeg: 225,

    /**
     * Elevation angle above the water-level (horizontal):
     * 0° = looking along the horizon, 90° = straight up
     */
    elevationDeg: 20,

    /**
     * Cone half-angle (i.e., half the FOV) in degrees.
     * Example: a 60° FOV camera has half-angle = 30°.
     */
    coneHalfAngleDeg: FOV.half.diagonalDeg,
};

/* ---------- Geometry helpers ---------- */

/**
 * Given a cone with apex at the camera and axis pointing with (azimuth,elevation),
 * the cross-section at height h above the camera (horizontal plane) is an ellipse.
 *
 * We approximate it as:
 * - axial distance along the camera axis to reach height h: t = h / sin(e)
 * - radius of the cone at that axial distance: r = t * tan(alpha)
 * - projected ellipse on the horizontal plane:
 *     semi-minor b ≈ r
 *     semi-major a ≈ r / cos(e)
 * - ellipse center is forward along azimuth by d = t * cos(e) from the camera position.
 *
 * This is a good visual approximation for mapping.
 */

export type EllipseParams = {
    center: LatLng;     // ellipse center on the map
    semiMajorM: number; // meters
    semiMinorM: number; // meters
    bearingDeg: number; // orientation of major axis (same as azimuth)
};

export function ellipseAtHeightMeters(
    h: number,
    {
        azimuthDeg = CAMERA.azimuthDeg,
        elevationDeg = CAMERA.elevationDeg,
        coneHalfAngleDeg = CAMERA.coneHalfAngleDeg,
        origin = CAMERA.position,
    }: {
        azimuthDeg?: number;
        elevationDeg?: number;
        coneHalfAngleDeg?: number;
        origin?: LatLng;
    } = {}
): EllipseParams {
    const e = deg2rad(elevationDeg);
    const a = deg2rad(coneHalfAngleDeg);

    if (Math.sin(e) === 0) {
        // Looking exactly horizontal → the horizontal plane at +h never intersects.
        // Return a tiny degenerate ellipse at origin to avoid NaNs.
        return {
            center: origin,
            semiMajorM: 0,
            semiMinorM: 0,
            bearingDeg: azimuthDeg,
        };
    }

    // Distance along the axis to reach height h
    const t = h / Math.sin(e);

    // Cone radius at that axial distance
    const r = t * Math.tan(a);

    // Projected ellipse dimensions on the horizontal plane (approx)
    const semiMinorM = r;
    const semiMajorM = r / Math.cos(e); // stretch due to tilt

    // Center offset on ground plane (forward along azimuth)
    const forward = t * Math.cos(e); // meters
    const center = moveMeters(origin, forward, azimuthDeg);

    return {
        center,
        semiMajorM,
        semiMinorM,
        bearingDeg: azimuthDeg,
    };
}

/**
 * Generate polygon points (LatLng[]) approximating an ellipse.
 * - center: WGS84 center
 * - aM / bM: semi-major / semi-minor in meters
 * - bearingDeg: direction of major axis, degrees from North
 * - n: number of points
 */
export function ellipsePolygon(
    center: LatLng,
    aM: number,
    bM: number,
    bearingDeg: number,
    n = 128
): LatLng[] {
    const pts: LatLng[] = [];
    const theta0 = deg2rad(bearingDeg);

    for (let i = 0; i < n; i++) {
        const t = (i / n) * 2 * Math.PI;
        // point in ellipse local coords (x along major, y along minor)
        const x = aM * Math.cos(t);
        const y = bM * Math.sin(t);

        // rotate by bearing
        const xr = x * Math.cos(theta0) - y * Math.sin(theta0);
        const yr = x * Math.sin(theta0) + y * Math.cos(theta0);

        // convert local meters to lat/lng
        pts.push(offsetMeters(center, xr, yr));
    }
    return pts;
}

/* ---------- Small geodesy helpers (local planar approximation) ---------- */

const METERS_PER_DEG_LAT = 111_320; // average

function metersPerDegLon(latDeg: number) {
    return METERS_PER_DEG_LAT * Math.cos(deg2rad(latDeg));
}

function offsetMeters(origin: LatLng, dxEastM: number, dyNorthM: number): LatLng {
    const dLat = dyNorthM / METERS_PER_DEG_LAT;
    const dLng = dxEastM / metersPerDegLon(origin.lat);
    return { lat: origin.lat + dLat, lng: origin.lng + dLng };
}

function moveMeters(origin: LatLng, distanceM: number, bearingDeg: number): LatLng {
    const b = deg2rad(bearingDeg);
    const dx = distanceM * Math.sin(b); // east
    const dy = distanceM * Math.cos(b); // north
    return offsetMeters(origin, dx, dy);
}

function deg2rad(d: number) {
    return (d * Math.PI) / 180;
}