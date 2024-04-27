{{content}}

{#if notes}
<notes>
  {#each notes as note}
    <note>
      <id>{{note.id}}</id>
      <source>{{note.source}}</source>
      <title>{{note.title}}</title>
      <summary>{{note.summary}}</summary>
      <complete>{{note.complete}}</complete>
    </note>
  {/each}
</notes>
{/if}
