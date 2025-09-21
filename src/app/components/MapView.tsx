"use client";

import dynamic from "next/dynamic";
import { ObjectUpdate } from "@/lib/database/schema";

// Dynamically import MapContainer so it only runs in browser
const LeafletMap = dynamic(() => import("./RealMapView"), { ssr: false });

export default function MapView({ objects }: { objects: ObjectUpdate[] }) {
  return <LeafletMap objects={objects} />;
}
