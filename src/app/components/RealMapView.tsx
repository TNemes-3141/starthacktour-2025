"use client";

import { MapContainer, TileLayer, Polygon, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { CAMERA, ellipseAtHeightMeters, ellipsePolygon } from "@/lib/camera";

L.Icon.Default.mergeOptions({
    iconRetinaUrl: "/marker-icon-2x.png",
    iconUrl: "/marker-icon.png",
    shadowUrl: "/marker-shadow.png",
});

export default function RealMapView() {
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
        </MapContainer>
    );
}
