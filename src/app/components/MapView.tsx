"use client";

import dynamic from "next/dynamic";

// Dynamically import MapContainer so it only runs in browser
const LeafletMap = dynamic(() => import("./RealMapView"), { ssr: false });

export default function MapView() {
  return <LeafletMap />;
}
