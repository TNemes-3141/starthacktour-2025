"use client";

import { MapContainer, TileLayer, Circle, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

L.Icon.Default.mergeOptions({
    iconRetinaUrl: "/marker-icon-2x.png",
    iconUrl: "/marker-icon.png",
    shadowUrl: "/marker-shadow.png",
});

export default function RealMapView() {
    const center: [number, number] = [46.2044, 6.1432];

    return (
        <MapContainer center={center} zoom={13} className="h-full w-full" scrollWheelZoom>
            <TileLayer
                attribution='&copy; <a href="https://osm.org/copyright">OSM</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <Circle center={center} radius={500} color="red" />
            <Marker position={center}>
                <Popup>Center of Geneva</Popup>
            </Marker>
        </MapContainer>
    );
}
