# Contributing

Open a pull request against `main` and discuss changes with the maintainers here.
When the contribution is ready for internal review, a maintainer adds the
`import` label. A bot imports the patch into Fireworks' authoritative source;
updates to your PR refresh that import and require another review.

After review, the normal cookbook promotion publishes the changes. The bot
then closes your original PR as incorporated and links to the public commit.
Your commit authorship and existing co-author credit are preserved through
`Co-authored-by` attribution in the bot-authored commit that first promotes your
contribution. Later promotion commits do not repeat that credit. Maintainers
merge generated promotion PRs with **Create a merge commit** to preserve these
individual commits and their attribution. The public repository disables squash
and rebase merges so these commit boundaries cannot be discarded accidentally.
Use a GitHub-associated commit email or your GitHub noreply address for profile
credit. You do not need access to the internal repository.

The import checks run periodically, so changes may take about ten minutes to
appear. Public contribution PRs remain open for review; their changes reach
`main` through the generated promotion PR.
