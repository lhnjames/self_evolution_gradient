import { cn } from "@/lib/utils";
import { ExternalLink } from "lucide-react";
import Image from "next/image";
import type { ReactNode } from "react";

interface ModalCalloutProps {
  title: string;
  children: ReactNode;
  eyebrow?: string;
  className?: string;
}

export function ModalCallout({
  title,
  children,
  eyebrow = "Modal cloud execution",
  className,
}: ModalCalloutProps) {
  return (
    <aside
      className={cn(
        "not-prose overflow-hidden rounded-xl border border-emerald-500/20 bg-emerald-500/[0.04]",
        className,
      )}
    >
      <div className="grid gap-5 p-5 sm:grid-cols-[12rem_1fr] sm:items-center sm:p-6">
        <a
          href="https://modal.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex min-h-20 items-center justify-center rounded-lg bg-[#0C0F0B] px-4 py-5 transition-transform hover:scale-[1.01]"
          aria-label="Visit Modal"
        >
          <Image
            src="/partners/modal-logo.svg"
            alt="Modal"
            width={187}
            height={36}
            className="h-auto w-full max-w-[187px]"
          />
        </a>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700 dark:text-emerald-400">
            {eyebrow}
          </p>
          <h3 className="text-lg font-semibold tracking-tight text-foreground">
            {title}
          </h3>
          <div className="mt-2 text-sm leading-relaxed text-muted-foreground">
            {children}
          </div>
          <a
            href="https://modal.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-foreground underline-offset-4 hover:underline"
          >
            Learn about Modal
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          </a>
        </div>
      </div>
    </aside>
  );
}
