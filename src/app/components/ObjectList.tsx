"use client";

import { useEffect, useState } from "react";
import EmptyState from "./EmptyState";
import { ObjectData } from "./types";
import { Card, CardHeader, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { Divider } from "@heroui/divider";
import { Accordion, AccordionItem } from "@heroui/accordion";
import { Image } from "@heroui/image";
import { ScrollShadow } from "@heroui/scroll-shadow";

interface Props {
  title?: string;
  type: "active" | "past";
}

export default function ObjectList({ title, type }: Props) {
  const [objects, setObjects] = useState<ObjectData[]>([
    {
      id: 1,
      timestamp: new Date("2025-01-01T12:00:00Z"),
      bbox: {
        top: 0,
        left: 0,
        bottom: 0,
        right: 0,
      },
      class: "Paraglider",
      confidence: 0.76,
      speed_mps: 3.41,
      distance: 244,
      latitude: 46.2044,
      longitude: 6.1432,
    }
  ]);

  /*useEffect(() => {
    fetchLocations().then((data) => {
      const filtered = data.filter((obj) =>
        type === "active" ? isActive(obj.timestamp) : !isActive(obj.timestamp)
      );
      setObjects(filtered);
    });
  }, [type]);*/

  return (
    <div className="flex flex-col h-full">
      {title ? <div className="border-b-1.5 border-b-gray-200 px-4 py-2 text-lg">{title}</div> : <></>}
      <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-white">
        {objects.length === 0 ? (
          <EmptyState text={`No ${type} objects yet.`} />
        ) : (
          objects.map((obj) => (
            <Card key={obj.id}>
              <CardHeader className="flex justify-between items-end">
                <div className="flex gap-3 items-center">
                  <p className="text-xl font-bold">#{obj.id}: {obj.class}</p>
                  <p className="text-md text-gray-400">{(obj.confidence * 100).toFixed(0)} %</p>
                </div>
                <Chip>{formatDate(obj.timestamp)}</Chip>
              </CardHeader>
              <Divider />
              <CardBody className="flex flex-col gap-3">
                <div className="grid grid-cols-4 text-center gap-4">
                  {/* Speed */}
                  <div>
                    <div className="font-semibold text-sm">Speed (m/s)</div>
                    <div>{obj.speed_mps.toFixed(2)}</div>
                  </div>

                  {/* Distance */}
                  <div>
                    <div className="font-semibold text-sm">Distance to cam (m)</div>
                    <div>{obj.distance.toFixed(1)}</div>
                  </div>

                  {/* Latitude */}
                  <div>
                    <div className="font-semibold text-sm">Latitude</div>
                    <div>{obj.latitude.toFixed(5)}</div>
                  </div>

                  {/* Longitude */}
                  <div>
                    <div className="font-semibold text-sm">Longitude</div>
                    <div>{obj.longitude.toFixed(5)}</div>
                  </div>
                </div>
                <Accordion>
                  <AccordionItem key="1" aria-label="Image details" title="Image details">
                    <div className="flex justify-center">
                      <Image
                        alt="HeroUI hero Image"
                        src="https://szctehfhcijgnwwlvvco.supabase.co/storage/v1/object/public/images/4091e73a-2df1-4912-a939-594d32695c4f/2025-09-20/snapshot_2025-09-20T10-00-51.389602.jpg"
                        width={400}
                      />
                    </div>

                  </AccordionItem>
                </Accordion>
              </CardBody>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}

function formatDate(date: Date): string {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  return `${yyyy}/${mm}/${dd} ${hh}:${min}:${ss}${ms}`;
}
