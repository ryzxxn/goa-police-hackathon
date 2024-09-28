import { useState, useEffect } from "react";
import { IoIosDocument } from "react-icons/io";
import { Supabase } from "../../../utils/SupabaseConnector";

export default function page() {
  return (
    <>
    <div className='flex flex-1 w-full h-screen'>
        <div className='flex bg-green-200 p-4'>
            <div>
            <IoIosDocument />
            </div>
        </div>
        <div className='flex bg-red-300'>

        </div>
    </div>
    </>
  )
}
