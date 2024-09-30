"use client";
import React, { useState, useEffect } from 'react';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { useParams } from 'next/navigation';
import { AreaChart, Area, XAxis, YAxis } from 'recharts';

const supabaseUrl = 'https://dlmwlgnyehclzrryxepq.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsbXdsZ255ZWhjbHpycnl4ZXBxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzQ0OTQ0OCwiZXhwIjoyMDQzMDI1NDQ4fQ.b4plF-vw8ZJ5g-E84LWwUMF5OEzE-NBuG-9sI4sw8BE'; // Ensure to keep your keys secure
const supabase: SupabaseClient = createClient(supabaseUrl, supabaseKey);

export default function Page() {
    const { id } = useParams();
    const [reportData, setReportData] = useState<any[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
  
    useEffect(() => {
        const fetchReportData = async () => {
            setLoading(true);
            const { data, error } = await supabase
                .from('gp_reports')
                .select("*")
                .eq('report_id', id);
  
            if (data) {
                setReportData(data);
            } else if (error) {
                console.error("Error fetching report data:", error);
            }
            setLoading(false);
        };
  
        fetchReportData();
    }, [id]);
  
    // Handle loading state
    if (loading) {
        return <div>Loading...</div>;
    }

    // Parse frame_timeline
    const frameTimeline = reportData.length > 0 && reportData[0].frame_timeline ? JSON.parse(reportData[0].frame_timeline) : [];
    const frameCount = frameTimeline.length;

    // Prepare data for chart
    const chartData = frameTimeline.map((item: any, index: number) => ({
        frame: index,
        score: item.score,
    }));
  
    // Calculate score for display
    const score = reportData.length > 0 && reportData[0].score !== null ? reportData[0].score * 10 : 0;
  
    // Determine threat level
    const threatLevel = score < 40 ? 'Low Risk' : score < 70 ? 'Moderate Risk' : 'High Risk';
  
    const handlePrint = () => {
        window.print();
    };

    return (
        <div className="p-4">
            <button onClick={handlePrint} className="mb-4 px-4 py-2 bg-blue-500 text-white rounded">
                Print Report
            </button>
            {reportData.length > 0 ? (
                <>
                    <div>
                        <div className='flex justify-between'>
                            <div>
                                <h1 className="text-xl font-bold">Overall Safety Score</h1>
                                {score !== null && (
                                    <div className="text-5xl font-bold mb-2">{score}%</div>
                                )}
                            </div>
                            <div className='text-red-500 font-extrabold text-[3rem]'>
                                RADWATCH
                            </div>
                        </div>
  
                        {score !== null && (
                            <div className="relative w-full h-4 bg-black rounded">
                                <div
                                    className="absolute h-full bg-red-500 rounded"
                                    style={{ width: `${score}%` }}
                                ></div>
                            </div>
                        )}
                        <p className={`text-sm ${score < 40 ? 'text-green-500' : score < 70 ? 'text-yellow-500' : 'text-red-500'}`}>
                            {threatLevel} Detected
                        </p>
                    </div>
                    
                    {reportData[0].type !== "text" && (
                        <div className="p-4 flex flex-1 w-full">
                            <div className="flex flex-1 flex-col">
                                {frameCount > 0 && (
                                    <>
                                        <h2 className="text-xl font-bold">Frame Count</h2>
                                        <p>Total Frames: {frameCount}</p>
                                    </>
                                )}
                            </div>

                            {chartData.length > 0 && (
                                <div className='flex flex-1 flex-col'>
                                    <h2 className="text-xl font-bold">Frame Analysis</h2>
                                    <AreaChart
                                        width={600}
                                        height={300}
                                        data={chartData}
                                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                    >
                                        <XAxis dataKey="frame" />
                                        <YAxis 
                                            domain={[0, 'dataMax']}
                                            ticks={[0,1,2,3,4,5,6,7,8,9,10]} // 10 ticks from 0 to 10
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="score"
                                            stroke="#ff0000"
                                            fill="#ff0000"
                                            fillOpacity={0.3}
                                        />
                                    </AreaChart>
                                </div>
                            )}
                        </div>
                    )}

                    <div>
                        <h1 className='font-bold'>Summary</h1>
                        {reportData[0].summary && (
                            <p className='font-mono'>{reportData[0].summary}</p>
                        )}
                    </div>
                </>
            ) : (
                <div>No report data available.</div>
            )}
        </div>
    );
}