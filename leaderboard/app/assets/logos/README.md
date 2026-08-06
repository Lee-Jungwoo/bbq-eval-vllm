# Family logos

Drop vendor logo files here and the leaderboard ranking table picks them up on
the next rerun. Nothing is committed to this directory: vendor logos are
trademarks, they are not ours to redistribute, and each vendor sets its own
terms for use of the mark. Fetch them from the vendor's own brand page and
check those terms before publishing the app anywhere public.

`app/lib/logos.py` maps a model family to a file stem. Expected names:

| file          | families it serves              | source                                    |
|---------------|---------------------------------|-------------------------------------------|
| `gemma.svg`   | Gemma                           | Google brand resources / the Gemma model card |
| `openai.svg`  | GPT-OSS                         | OpenAI brand guidelines                   |
| `qwen.svg`    | Qwen2.5, Qwen3, Qwen3.5         | Alibaba Qwen brand assets                 |
| `meta.svg`    | Llama                           | Meta brand resources                      |
| `mistral.svg` | Ministral-3, Mistral            | Mistral AI brand assets                   |
| `microsoft.svg` | Phi                           | Microsoft brand resources                 |

`.png` also works if no SVG is available — `.svg` is tried first.

A family with no file simply renders no logo; the model name column is
unaffected. Adding a new family means adding a row to `LOGO_STEMS` in
`app/lib/logos.py`.

Keep the artwork small and roughly square. The table renders it at row height,
so anything wider than ~4:1 will be unreadable.
