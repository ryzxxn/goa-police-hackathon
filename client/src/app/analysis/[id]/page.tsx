"use client"
import React from 'react'
import { useParams } from 'next/navigation'
import { Progress } from "@/components/ui/progress"
import { Area, AreaChart, XAxis, YAxis } from "recharts"
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { useState, useEffect } from 'react'



import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

import { PolarAngleAxis, RadialBar, RadialBarChart } from "recharts"
import PDFDownloadButton from '../../pdfDownloader/pdfDownloader';


export default function page() {
    const {id} = useParams()
     useEffect(() => {
        fetchReportData()
     },[id])

     const [reportData,setReportData] = useState<any[]>([])

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

const pieChartData = [
    { name: 'Safe', value: 60 },
    { name: 'Questionable', value: 25 },
    { name: 'Dangerous', value: 15 },
]

const COLORS = ['#10B981', '#FBBF24', '#EF4444']

const score = reportData.length > 0 ? <>{reportData[0].score*10}%</> : null;
const framesProcessed = reportData.length > 0 && reportData[0].frame_timeline === null? <>asdas</> : null;
const summary = "text that you want to be in the pdf" 
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
                            <Progress value = {score} className="h-2 mb-2 bg-gray-500" />
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
                    { reportData.length > 0 && reportData[0].frame_timeline !== null && 
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
                                             value: (reportData[0].frame_timeline[i].score ),  
                                             frames: i,     
                                         }))}
                                        margin={{ left: 0, right: 0, top: 0, bottom: 0 }}
                                    >
                                        <XAxis dataKey="score"  />
                                        <YAxis domain={["dataMin ", "dataMax "]}  />
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
                                                        { value }
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

                    <Card className="bg-gray-950 shadow-md flex gap-8 w-1/2 border-white border-2">
                        <CardContent className="flex gap-4 p-4">
                            <div className="grid items-center gap-2">
                                <div className="grid flex-1 auto-rows-min gap-0.5">
                                    <div className="text-sm text-muted-foreground">Move</div>
                                    <div className="flex items-baseline gap-1 text-xl font-bold tabular-nums leading-none text-gray-100">
                                        562/600
                                        <span className="text-sm font-normal text-muted-foreground text-gray-400">
                                            kcal
                                        </span>
                                    </div>
                                </div>
                                <div className="grid flex-1 auto-rows-min gap-0.5">
                                    <div className="text-sm text-muted-foreground">Exercise</div>
                                    <div className="flex items-baseline gap-1 text-xl font-bold tabular-nums leading-none text-gray-100">
                                        73/120
                                        <span className="text-sm font-normal text-muted-foreground text-gray-400">
                                            min
                                        </span>
                                    </div>
                                </div>
                                <div className="grid flex-1 auto-rows-min gap-0.5">
                                    <div className="text-sm text-muted-foreground">Stand</div>
                                    <div className="flex items-baseline gap-1 text-xl font-bold tabular-nums leading-none text-gray-100">
                                        8/12
                                        <span className="text-sm font-normal text-muted-foreground text-gray-400">
                                            hr
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <ChartContainer
                                config={{
                                    move: {
                                        label: "Move",
                                        color: "hsl(var(--chart-1))",
                                    },
                                    exercise: {
                                        label: "Exercise",
                                        color: "hsl(var(--chart-2))",
                                    },
                                    stand: {
                                        label: "Stand",
                                        color: "hsl(var(--chart-3))",
                                    },
                                }}
                                className="mx-auto aspect-square w-full max-w-[80%]"
                            >
                                <RadialBarChart
                                    margin={{
                                        left: -10,
                                        right: -10,
                                        top: -10,
                                        bottom: -10,
                                    }}
                                    data={[
                                        {
                                            activity: "stand",
                                            value: (8 / 12) * 100,
                                            fill: "var(--color-stand)",
                                        },
                                        {
                                            activity: "exercise",
                                            value: (46 / 60) * 100,
                                            fill: "var(--color-exercise)",
                                        },
                                        {
                                            activity: "move",
                                            value: (245 / 360) * 100,
                                            fill: "var(--color-move)",
                                        },
                                    ]}
                                    innerRadius="20%"
                                    barSize={24}
                                    startAngle={90}
                                    endAngle={450}
                                >
                                    <PolarAngleAxis
                                        type="number"
                                        domain={[0, 100]}
                                        dataKey="value"
                                        tick={false}
                                    />
                                    <RadialBar dataKey="value" background cornerRadius={5} />
                                </RadialBarChart>
                            </ChartContainer>
                        </CardContent>
                    </Card>



                </div>


                <Card className="mt-8 shadow-lg bg-gray-950 border-white border-2">
                    <CardHeader>
                        <CardTitle className="text-xl font-semibold text-gray-100">Summary Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-gray-300 space-y-4">
                            <p>
                                Based on the recent content analysis, we've identified several areas of concern across different media types.
                                Out of 152 pieces of content analyzed, 23 items have been flagged for further review.
                            </p>
                            <p>
                                The primary issues detected include:
                            </p>
                            <ul className="list-disc list-inside space-y-2 pl-4">
                                <li>Potential hate speech in video content, particularly in political contexts</li>
                                <li>Extremist symbols appearing in image-based media, such as protest materials</li>
                                <li>Discriminatory language in text-based forum posts and online discussions</li>
                                <li>Misinformation spread through social media memes and images</li>
                                <li>Violent content in some video interviews and reports</li>
                                <li>Conspiracy theories propagated in blog articles and longer-form text content</li>
                            </ul>
                            <p>
                                The overall safety score of 72% indicates a moderate level of risk. While the majority of content appears
                                safe, the presence of these flagged items suggests a need for ongoing monitoring and potentially more
                                stringent content guidelines.
                            </p>
                            <p>
                                It is recommended to conduct a more detailed review of the flagged content, particularly focusing on the
                                sources and contexts of the most severe violations. This will help in developing more targeted strategies
                                for content moderation and user education.
                            </p>
                        </div>
                    </CardContent>
                </Card>

                <div className="mt-8 flex justify-end">
                        <PDFDownloadButton text={summary} />
                </div>
            </div>
        </div >
        </div>
    
  )
}
