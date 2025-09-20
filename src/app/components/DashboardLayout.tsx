interface Props {
    leftTop: React.ReactNode;
    leftBottom: React.ReactNode;
    right: React.ReactNode;
  }
  
  export default function DashboardLayout({ leftTop, leftBottom, right }: Props) {
    return (
      <div className="flex h-[calc(100vh-4rem)]"> {/* subtract header */}
        {/* Left Panel */}
        <div className="w-1/2 flex flex-col border-r border-gray-200">
          <div className="flex-1 overflow-hidden">{leftTop}</div>
          <div className="flex-1 overflow-hidden">{leftBottom}</div>
        </div>
  
        {/* Right Panel */}
        <div className="w-1/2 relative">{right}</div>
      </div>
    );
  }
  