// app/components/ObjectCard.tsx
"use client";

import { Card, CardHeader, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { Divider } from "@heroui/divider";
import { Accordion, AccordionItem } from "@heroui/accordion";
import { Image } from "@heroui/image";
import { ObjectUpdate } from "@/lib/database/schema";
import { useRelativeTime } from "./useRelativeTime";
import { cn } from "@/lib/utils";

interface Props {
  obj: ObjectUpdate;
  variant?: "static" | "relative";
}

export default function ObjectCard({ obj, variant = "static" }: Props) {
  const timestamp = new Date(obj.timestamp);
  const relative = useRelativeTime(timestamp);

  return (
    <Card key={obj.objectId}>
      <CardHeader className="flex justify-between items-end">
        <div className="flex gap-3 items-center">
          <p className="text-xl font-bold">
            #{obj.objectId}: {obj.class}
          </p>
          <p className="text-md text-gray-400">
            {(obj.confidence * 100).toFixed(0)} %
          </p>
        </div>
        {variant === "static" ? (
          <Chip>{formatDate(timestamp)}</Chip>
        ) : (
          <Chip>{relative}</Chip>
        )}
      </CardHeader>
      <Divider />
      <CardBody className="flex flex-col gap-3">
        <div className={cn("grid text-center gap-4", variant === "static" ? "grid-cols-5" : "grid-cols-4")}>
          {/* Speed */}
          <div>
            <div className="font-bold text-sm">Velocity</div>
            <div>{obj.speedMps?.toFixed(2) ?? "N.A."} m/s</div>
          </div>

          {/* Distance */}
          <div>
            <div className="font-bold text-sm">Distance to cam</div>
            <div>{obj.distanceM?.toFixed(1) ?? "N.A."} m</div>
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

          {variant === "static" && (
            <div>
              <div className="font-bold text-sm">Timestamp</div>
              <div>{formatDate(timestamp)}</div>
            </div>
          )}
        </div>

        <Accordion>
          <AccordionItem key="1" aria-label="Image details" title="Image details">
            <div className="flex justify-center">
              <Image
                alt="Snapshot"
                src={obj.snapshotUrl ?? ""}
                width={400}
              />
            </div>
          </AccordionItem>
        </Accordion>
      </CardBody>
    </Card>
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