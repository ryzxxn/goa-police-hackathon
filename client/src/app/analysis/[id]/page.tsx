"use client"
import React from 'react'
import { useParams } from 'next/navigation'
import { Progress } from "@/components/ui/progress"
import { Area, AreaChart, XAxis, YAxis } from "recharts"
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { useState, useEffect } from 'react'

import { Card, CardContent, CardHeader, CardDescription, CardTitle } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

import PDFDownloadButton from '../../pdfDownloader/pdfDownloader';


export default function page() {
    const { id } = useParams()
    useEffect(() => {
        fetchReportData()
    }, [id])

    const [reportData, setReportData] = useState<any[]>([])

    async function fetchReportData() {

        let { data, error } = await supabase
            .from('gp_reports')
            .select("*")

            // Filters
            .eq('report_id', id)

        if (data) {
            setReportData(data)
        }

    }

    console.log(reportData);


    const supabaseUrl = 'https://dlmwlgnyehclzrryxepq.supabase.co';
    const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsbXdsZ255ZWhjbHpycnl4ZXBxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzQ0OTQ0OCwiZXhwIjoyMDQzMDI1NDQ4fQ.b4plF-vw8ZJ5g-E84LWwUMF5OEzE-NBuG-9sI4sw8BE';
    const supabase: SupabaseClient = createClient(supabaseUrl, supabaseKey);

    //////
    const COLOR_RANGES = [
        { min: 0, max: 4, color: '#4ade80' },  // green
        { min: 5, max: 7, color: '#facc15' },  // yellow
        { min: 8, max: 10, color: '#f87171' }, // red
    ]

    // Function to categorize and count scores
    const categorizeCounts = (data: { score: number }[]) => {
        const counts = COLOR_RANGES.map(range => ({
            name: `${range.min}-${range.max}`,
            value: 0,
            color: range.color
        }))

        data.forEach(item => {
            const score = item.score
            const rangeIndex = COLOR_RANGES.findIndex(range => score >= range.min && score <= range.max)
            if (rangeIndex !== -1) {
                counts[rangeIndex].value++
            }
        })

        return counts.filter(item => item.value > 0)
    }

    interface ScorePieChartProps {
        frameTimeline: { score: number }[]
    }
    const frameTimeline = [
        { score: 2 },
        { score: 3 },
        { score: 5 },
        { score: 6 },
        { score: 8 },
        { score: 9 },
        // ... more data
    ]
    ///////


    const score = reportData.length > 0 ? <>{reportData[0].score * 10}%</> : null;
    const framesProcessed = reportData.length > 0 && reportData[0].frame_timeline === null ? <>asdas</> : null;
    const report_summary = "text that you want to be in the pdf"
    const chartData = categorizeCounts(frameTimeline)
    return (
        <div className="relative">
            <div className="inset-0 min-h-full bg-fixed bg-gradient-to-b from-gray-900   to-red-800 text-gray-100 p-8">
                <div className=" max-w-7xl mx-auto">
                    <h1 className="text-3xl font-bold mb-8">Content Analysis Dashboard</h1>

                    <div className=" grid gap-8 md:grid-cols-2 lg:grid-cols-2 ">
                        <Card className="bg-gray-950 border-white border-2">
                            <CardHeader>
                                <CardTitle className="text-xl font-semibold text-gray-100">Overall Safety Score</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-5xl font-bold mb-2 text-gray-100">{score}</div>
                                <Progress value={score} className="h-2 mb-2 bg-gray-500" />
                                <p className="text-sm text-gray-400">Moderate risk detected</p>
                            </CardContent>
                        </Card>

                        <Card className="bg-gray-950 border-white border-2">
                            <CardHeader>
                                <CardTitle className="text-xl font-semibold text-gray-100">Analyzed Content</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-5xl font-bold mb-2 text-gray-100">{framesProcessed}</div>
                                <p className="text-sm text-gray-400">Total pieces of content analyzed</p>
                            </CardContent>
                        </Card>

                    </div>

                    <div className="py-16 flex flex-1 justify-evenly space-x-[5rem] ">
                        <div className=" flex gap-8 w-1/2">
                            {reportData.length > 0 && reportData[0].frame_timeline !== null &&
                                <Card className="px-6 shadow-md bg-gray-950 border-white border-2 flex-1  ">
                                    <CardHeader className="space-y-0 pb-0 text-gray-800">
                                        <CardDescription >Total number of frames </CardDescription>
                                        <CardTitle className=" flex items-baseline gap-1 text-m tabular-nums text-gray-100">
                                            Frames
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        <ChartContainer
                                            config={{
                                                frames: {
                                                    label: "Frames",
                                                    color: "red",
                                                },
                                            }}
                                        >
                                            <AreaChart
                                                accessibilityLayer
                                                data={Array.from({ length: reportData[0].frame_timeline.length }, (_, i) => ({
                                                    value: (reportData[0].frame_timeline[i].score),
                                                    frames: i,
                                                }))}
                                                margin={{ left: 0, right: 0, top: 0, bottom: 0 }}
                                            >
                                                <XAxis dataKey="score" />
                                                <YAxis domain={["dataMin ", "dataMax "]} />
                                                <defs>
                                                    <linearGradient id="fillFrames" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="white" stopOpacity={0.8} />
                                                        <stop offset="95%" stopColor="red" stopOpacity={0.1} />
                                                    </linearGradient>
                                                </defs>
                                                <Area
                                                    dataKey="frames"
                                                    type="natural"
                                                    fill="url(#fillFrames)"
                                                    fillOpacity={0.4}
                                                    stroke="white"
                                                />
                                                <ChartTooltip
                                                    cursor={false}
                                                    content={<ChartTooltipContent hideLabel />}
                                                    formatter={(value) => (
                                                        <div className="flex min-w-[120px] items-center text-xs text-muted-foreground text-gray-950">
                                                            Total number of frames
                                                            <div className="ml-auto flex items-baseline gap-0.5 font-mono font-medium tabular-nums text-foreground">
                                                                {value}
                                                                <span className="font-normal text-muted-foreground"> frames </span>
                                                            </div>
                                                        </div>
                                                    )}
                                                />
                                            </AreaChart>
                                        </ChartContainer>
                                    </CardContent>
                                </Card>
                            }

                        </div>

                        <div className=' px-6 shadow-md bg-gray-950 border-white border-2 flex-1'>
                            <PieChart width={400} height={400}>
                                <Pie
                                    data={chartData}
                                    cx="60%"
                                    cy="50%"
                                    labelLine={false}
                                    outerRadius={150}
                                    fill="#8884d8"
                                    dataKey="value"
                                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                >
                                    {chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip />
                                <Legend />
                            </PieChart>
                        </div>



                    </div>



                    <Card className="mt-8 shadow-lg bg-gray-950 border-white border-2">
                        <CardHeader>
                            <CardTitle className="text-xl font-semibold text-gray-100">Summary Analysis</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-gray-300 space-y-4">
                                <p>
                                    {reportData.length > 0 && reportData[0].summary}
                                </p>
                            </div>
                        </CardContent>
                    </Card>

                    <div className="mt-8 flex justify-end">
                        <PDFDownloadButton text={report_summary} />
                    </div>
                </div>
            </div >
        </div>

    )
}