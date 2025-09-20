"use client";

import EmptyState from "./EmptyState";
import { Card, CardHeader, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { Divider } from "@heroui/divider";
import { Accordion, AccordionItem } from "@heroui/accordion";
import { Image } from "@heroui/image";
import { ManagedObject } from "../useObjectManager";

interface Props {
  title?: string;
  type: "active" | "past";
  objects: ManagedObject[];
}

export default function ObjectList({ title, type, objects }: Props) {

  return (
    <div className="flex flex-col h-full">
      {title ? <div className="border-b-1.5 border-b-gray-200 px-4 py-2 text-lg">{title}</div> : <></>}
      <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-white">
        {objects.length === 0 ? (
          <EmptyState text={`No ${type} objects yet.`} />
        ) : (
          objects.map((obj) => (
            <Card key={obj.objectId}>
              <CardHeader className="flex justify-between items-end">
                <div className="flex gap-3 items-center">
                  <p className="text-xl font-bold">#{obj.objectId}: {obj.class}</p>
                  <p className="text-md text-gray-400">{(obj.confidence * 100).toFixed(0)} %</p>
                </div>
                <Chip>{formatDate(new Date(obj.timestamp))}</Chip>
              </CardHeader>
              <Divider />
              <CardBody className="flex flex-col gap-3">
                <div className="grid grid-cols-4 text-center gap-4">
                  {/* Speed */}
                  <div>
                    <div className="font-bold text-sm">Speed (m/s)</div>
                    <div>{obj.speedMps?.toFixed(2) ?? "N.A."}</div>
                  </div>

                  {/* Distance */}
                  <div>
                    <div className="font-bold text-sm">Distance to cam (m)</div>
                    <div>{obj.distanceM?.toFixed(1) ?? "N.A."}</div>
                  </div>

                  {/* Latitude */}
                  <div>
                    <div className="font-bold text-sm">Latitude</div>
                    <div>{obj.latitude?.toFixed(5) ?? "N.A."}</div>
                  </div>

                  {/* Longitude */}
                  <div>
                    <div className="font-bold text-sm">Longitude</div>
                    <div>{obj.longitude?.toFixed(5) ?? "N.A."}</div>
                  </div>
                </div>
                <Accordion>
                  <AccordionItem key="1" aria-label="Image details" title="Image details">
                    <div className="flex justify-center">
                      <Image
                        alt="HeroUI hero Image"
                        src={obj.snapshotUrl ?? ""}
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
