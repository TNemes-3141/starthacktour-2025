"use client";

import DashboardLayout from "./components/DashboardLayout";
import ObjectList from "./components/ObjectList";
import MapView from "./components/MapView";

export default function Home() {
  return (
    <DashboardLayout
      leftTop={<ObjectList title="Active Objects" type="active" />}
      leftBottom={<ObjectList title="Past Object Log" type="past" />}
      right={<MapView />}
    />
  );
}
