"use client"

import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Area, AreaChart, XAxis, YAxis } from "recharts"

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

const barChartData = [
    { name: 'Hate Speech', value: 65 },
    { name: 'Extremism', value: 45 },
    { name: 'Violence', value: 30 },
    { name: 'Discrimination', value: 50 },
    { name: 'Misinformation', value: 55 },
]

const pieChartData = [
    { name: 'Safe', value: 60 },
    { name: 'Questionable', value: 25 },
    { name: 'Dangerous', value: 15 },
]

const COLORS = ['#10B981', '#FBBF24', '#EF4444']

export default function DashboardAnalysis() {
    return (
        <div className="absolute inset-0 min-h-screen bg-gradient-to-b from-gray-900 to-red-600 text-gray-100 p-8">
            <div className=" max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold mb-8">Content Analysis Dashboard</h1>

                <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                    <Card className="bg-gray-950 border-gray-700">
                        <CardHeader>
                            <CardTitle className="text-xl font-semibold text-gray-100">Overall Safety Score</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-5xl font-bold mb-2 text-gray-100">72%</div>
                            <Progress value={72} className="h-2 mb-2 " />
                            <p className="text-sm text-gray-400">Moderate risk detected</p>
                        </CardContent>
                    </Card>

                    <Card className="bg-gray-950 border-gray-700">
                        <CardHeader>
                            <CardTitle className="text-xl font-semibold text-gray-100">Analyzed Content</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-5xl font-bold mb-2 text-gray-100">152</div>
                            <p className="text-sm text-gray-400">Total pieces of content analyzed</p>
                        </CardContent>
                    </Card>

                </div>

                <div className="py-16 flex flex-1 justify-evenly space-x-[5rem]">
                    <div className=" flex gap-8 w-1/2">
                        <Card className="bg-gray-950 border-gray-700 flex-1">
                            <CardHeader className="space-y-0 pb-0">
                                <CardDescription>Total number of frames:</CardDescription>
                                <CardTitle className="flex items-baseline gap-1 text-4xl tabular-nums text-gray-100">
                                    10
                                    <span className="font-sans text-sm font-normal tracking-normal text-muted-foreground">frames</span>
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="p-0">
                                <ChartContainer
                                    config={{
                                        frames: {
                                            label: "Frames",
                                            color: "hsl(var(--chart-2))",
                                        },
                                    }}
                                >
                                    <AreaChart
                                        accessibilityLayer
                                        data={Array.from({ length: 11 }, (_, i) => ({
                                            date: `2024-01-${i + 1}`,
                                            frames: i,
                                        }))}
                                        margin={{ left: 0, right: 0, top: 0, bottom: 0 }}
                                    >
                                        <XAxis dataKey="date" hide />
                                        <YAxis domain={["dataMin - 5", "dataMax + 2"]} hide />
                                        <defs>
                                            <linearGradient id="fillFrames" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="var(--color-frames)" stopOpacity={0.8} />
                                                <stop offset="95%" stopColor="var(--color-frames)" stopOpacity={0.1} />
                                            </linearGradient>
                                        </defs>
                                        <Area
                                            dataKey="frames"
                                            type="natural"
                                            fill="url(#fillFrames)"
                                            fillOpacity={0.4}
                                            stroke="var(--color-frames)"
                                        />
                                        <ChartTooltip
                                            cursor={false}
                                            content={<ChartTooltipContent hideLabel />}
                                            formatter={(value) => (
                                                <div className="flex min-w-[120px] items-center text-xs text-muted-foreground">
                                                    Total number of frames
                                                    <div className="ml-auto flex items-baseline gap-0.5 font-mono font-medium tabular-nums text-foreground">
                                                        {value}
                                                        <span className="font-normal text-muted-foreground">frames</span>
                                                    </div>
                                                </div>
                                            )}
                                        />
                                    </AreaChart>
                                </ChartContainer>
                            </CardContent>
                        </Card>
                    </div>

                    <Card className="bg-gray-950 flex gap-8 w-1/2">
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


                <Card className="mt-8 bg-gray-800 border-gray-700">
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
                    <Button className="bg-red-600 hover:bg-red-700 text-white">Generate Full Report</Button>
                </div>
            </div>
        </div >
    )
}