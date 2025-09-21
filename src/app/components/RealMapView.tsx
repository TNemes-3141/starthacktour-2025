"use client";

import { MapContainer, TileLayer, Polygon, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { CAMERA, ellipseAtHeightMeters, ellipsePolygon } from "@/lib/camera";
import { ObjectUpdate } from "@/lib/database/schema";

L.Icon.Default.mergeOptions({
    iconRetinaUrl: "/marker-icon-2x.png",
    iconUrl: "/marker-icon.png",
    shadowUrl: "/marker-shadow.png",
});

const warningDivIcon = L.divIcon({
    html: '<div style="color: red; font-size: 24px;">⚠️</div>',
    className: "", // clear Leaflet’s default styles
    iconSize: [24, 24],
    iconAnchor: [12, 12],
});

const parachuteDivIcon = L.divIcon({
    html: '<div style="color: red; font-size: 24px;">🪂</div>',
    className: "", // clear Leaflet’s default styles
    iconSize: [24, 24],
    iconAnchor: [12, 12],
});

const airplaneDivIcon = L.divIcon({
    html: '<div style="color: red; font-size: 24px;">🚁</div>',
    className: "", // clear Leaflet’s default styles
    iconSize: [24, 24],
    iconAnchor: [12, 12],
});

const birdDivIcon = L.divIcon({
    html: '<div style="color: red; font-size: 24px;">🦅</div>',
    className: "", // clear Leaflet’s default styles
    iconSize: [24, 24],
    iconAnchor: [12, 12],
});

const personDivIcon = L.divIcon({
    html: '<div style="color: red; font-size: 24px;">🧍🏻</div>',
    className: "", // clear Leaflet’s default styles
    iconSize: [24, 24],
    iconAnchor: [12, 12],
});

export default function RealMapView({ objects }: { objects: ObjectUpdate[] }) {
    const center: [number, number] = [CAMERA.position.lat, CAMERA.position.lng];

    // Compute ellipse (cone ∩ horizontal plane at +100m)
    const params = ellipseAtHeightMeters(100);
    const ellipsePts = ellipsePolygon(
        params.center,
        params.semiMajorM,
        params.semiMinorM,
        params.bearingDeg,
        180 // resolution
    );

    return (
        <MapContainer center={center} zoom={13} className="h-full w-full" scrollWheelZoom>
            <TileLayer
                attribution='&copy; <a href="https://osm.org/copyright">OSM</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* Camera position */}
            <Marker position={[CAMERA.position.lat, CAMERA.position.lng]}>
                <Popup>
                    <div className="text-sm">
                        <div><strong>Camera</strong></div>
                        <div>Azimuth: {CAMERA.azimuthDeg}°</div>
                        <div>Elevation: {CAMERA.elevationDeg}°</div>
                        <div>Half-FOV: {CAMERA.coneHalfAngleDeg}°</div>
                    </div>
                </Popup>
            </Marker>

            {/* Ellipse of the 100m slice */}
            <Polygon positions={ellipsePts.map((p) => [p.lat, p.lng] as [number, number])} pathOptions={{ color: "lightblue", weight: 2, fillOpacity: 0.12 }} />

            {objects
                .filter(
                    (o) =>
                        typeof o.latitude === "number" &&
                        typeof o.longitude === "number" &&
                        !isNaN(o.latitude) &&
                        !isNaN(o.longitude)
                )
                .map((o) => {
                    const scaled = anisotropicScaleFromCamera(
                        o.latitude!,
                        o.longitude!,
                        CAMERA.position.lat,
                        CAMERA.position.lng,
                        CAMERA.azimuthDeg,
                        5,  // stretch factor forward
                        10   // stretch factor sideways
                    );

                    return <WarningMarker
                        key={o.objectId}
                        id={o.objectId}
                        name={o.class}
                        confidence={o.confidence}
                        lat={scaled.lat}
                        lng={scaled.lng}
                        speedMps={o.speedMps ?? -1}
                        distanceM={o.distanceM ?? -1}
                    />
                })}
        </MapContainer>
    );
}

function WarningMarker({ lat, lng, id, name, confidence, speedMps, distanceM }: { lat: number; lng: number; id: string; name: string; confidence: number; speedMps: number, distanceM: number }) {
    var divIcon;

    switch (name) {
        case "paraglider":
            divIcon = parachuteDivIcon;
            break;
        case "person":
            divIcon = personDivIcon;
            break;
        case "airplane":
            divIcon = airplaneDivIcon;
            break;
        case "bird":
            divIcon = birdDivIcon;
            break;
        default:
            divIcon = warningDivIcon;
            break;
    }

    return (
        <Marker position={[lat, lng]} icon={divIcon}>
            <Popup>
                <div className="text-sm">
                    <div><strong>#{id}: {name} ({(confidence * 100).toFixed(0)}%)</strong></div>
                    <div>Velocity: {speedMps === -1 ? "N.A." : speedMps.toFixed(2)} m/s</div>
                    <div>Elevation: {distanceM == -1 ? "N.A." : distanceM.toFixed(2)} m</div>
                </div>
            </Popup>
        </Marker>
    );
}

const R = 6371000; // Earth radius in m

function latLngToMeters(lat: number, lng: number, refLat: number, refLng: number) {
    const dLat = (lat - refLat) * Math.PI / 180;
    const dLng = (lng - refLng) * Math.PI / 180;
    return {
        x: dLng * Math.cos(refLat * Math.PI / 180) * R,
        y: dLat * R,
    };
}

function metersToLatLng(x: number, y: number, refLat: number, refLng: number) {
    const dLat = y / R;
    const dLng = x / (R * Math.cos(refLat * Math.PI / 180));
    return {
        lat: refLat + dLat * 180 / Math.PI,
        lng: refLng + dLng * 180 / Math.PI,
    };
}

export function anisotropicScaleFromCamera(
    lat: number,
    lng: number,
    camLat: number,
    camLng: number,
    azimuthDeg: number,
    scaleForward: number,
    scalePerp: number
) {
    // Step 1: to local meters
    const { x, y } = latLngToMeters(lat, lng, camLat, camLng);

    // Step 2: rotate into camera frame
    const theta = azimuthDeg * Math.PI / 180;
    const xr = Math.cos(theta) * x + Math.sin(theta) * y;
    const yr = -Math.sin(theta) * x + Math.cos(theta) * y;

    // Step 3: anisotropic scaling
    const xrScaled = xr * scaleForward;
    const yrScaled = yr * scalePerp;

    // Step 4: rotate back
    const xFinal = Math.cos(theta) * xrScaled - Math.sin(theta) * yrScaled;
    const yFinal = Math.sin(theta) * xrScaled + Math.cos(theta) * yrScaled;

    // Step 5: back to lat/lng
    return metersToLatLng(xFinal, yFinal, camLat, camLng);
}
