import { ExternalLinkIcon } from "@radix-ui/react-icons";

export function EmptyLLMOutput() {
  return (
    <div className="md:mx-auto max-w-2xl md:px-4">
      <div className="rounded-lg border bg-background shadow-lg p-8">
        <h1 className="mb-2 text-lg font-semibold">
          Demo Chat with Function Calling Capabilities
        </h1>
        <div className="m-4 text-muted-foreground">
          This demo has been preconfigured with the following functions:
          <ul className="list-disc list-inside">
            <li>generate an image from a text description,</li>
            <li>render chart from numeric data,</li>
            <li>obtain last day's price of a given stock,</li>
            <li>get recent news articles related to a query.</li>
          </ul>
        </div>
        <div className="m-4 text-muted-foreground">
          Click on "show available functions" link above to see the function schema.
        </div>
        <div className="m-4 text-muted-foreground">
          It's easy to build an app with custom functions! The instructions are <a href="https://github.com/fw-ai/forge/tree/main/apps/functional_chat" className="text-blue-500 hover:text-blue-700">here <ExternalLinkIcon className="inline w-4 h-4" /></a>.
        </div>
      </div>
    </div>
  );
}
