export interface ObjectData {
    id: number;
    timestamp: Date;
    bbox: {
        top: number;
        left: number;
        bottom: number;
        right: number;
    }
    class: string;
    confidence: number;
    speed_mps: number;
    distance: number;
    latitude: number;
    longitude: number;
    image?: string;
}
