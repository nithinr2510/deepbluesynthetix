import Index from "./pages/Index";
import { Toaster } from "sonner";

function App() {
  return (
    <>
      <Toaster
        position="top-right"
        richColors
        theme="dark"
        toastOptions={{
          style: {
            background: "#111118",
            border: "1px solid #27272a",
            color: "#e4e4e7",
          },
        }}
      />
      <Index />
    </>
  );
}

export default App;
