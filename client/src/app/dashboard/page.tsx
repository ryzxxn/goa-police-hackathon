'use client'
import { useState, useEffect } from "react";
import { IoIosDocument } from "react-icons/io";
import { createClient } from "@supabase/supabase-js";
import Link from "next/link";

// Initialize Supabase client
const supabaseUrl = "https://dlmwlgnyehclzrryxepq.supabase.co";
const supabaseAnonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsbXdsZ255ZWhjbHpycnl4ZXBxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzQ0OTQ0OCwiZXhwIjoyMDQzMDI1NDQ4fQ.b4plF-vw8ZJ5g-E84LWwUMF5OEzE-NBuG-9sI4sw8BE";
const supabase = createClient(supabaseUrl, supabaseAnonKey);

export default function Dashboard() {
  const [reports, setReports] = useState<any>([]);
  const [sortOrder, setSortOrder] = useState("asc");

  useEffect(() => {
    const fetchReports = async () => {
      const { data, error } = await supabase
        .from("gp_reports")
        .select("*");
      
      if (error) {
        console.error("Error fetching reports:", error);
      } else {
        setReports(data);
      }
    };

    fetchReports();
  }, []);

  const formatDateTime = (dateTime: string) => {
    const date = new Date(dateTime);
    return `${String(date.getDate()).padStart(2, '0')}/${String(date.getMonth() + 1).padStart(2, '0')}/${date.getFullYear()} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  };

  const sortedReports = [...reports].sort((a, b) => {
    const dateA = new Date(a.created_at).getTime();
    const dateB = new Date(b.created_at).getTime();
    return sortOrder === "asc" ? dateA - dateB : dateB - dateA;
  });

  const toggleSortOrder = () => {
    setSortOrder(prev => (prev === "asc" ? "desc" : "asc"));
  };

  return (
    <div className='flex flex-col w-full h-screen bg-gradient-to-b from-gray-900 to-red-600'>
      <div className='flex items-center justify-between p-4 shadow-lg bg-white'>
        <div className='flex items-center'>
          <h1 className='ml-2 text-lg font-bold text-black'>Reports Dashboard</h1>
        </div>
        <button 
          className='px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition'
          onClick={toggleSortOrder}
        >
          Sort {sortOrder === "asc" ? "Descending" : "Ascending"}
        </button>
      </div>
      <div className='flex flex-col p-4 overflow-y-auto gap-4'>
        {sortedReports.map((report: any) => (
          <Link href={`/analysis/${report.report_id}`} key={report.id}>
            <div className='p-4 bg-white border-l-[1px] text-black'>
              <h2 className='font-semibold  text-black'>Report ID: {report.report_id}</h2>
              <p className='text-sm  text-black'>
                Created At: {formatDateTime(report.created_at)}
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}