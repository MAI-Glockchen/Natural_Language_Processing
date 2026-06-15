CREATE TABLE IF NOT EXISTS generated_articles (
    run_id bigserial PRIMARY KEY,
    article_id integer NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    split text NOT NULL,
    method text NOT NULL,
    prompt_version text NOT NULL,
    model_name text NOT NULL,
    top_k integer NOT NULL,
    topic text NOT NULL,
    index_file text NOT NULL,
    generated_title text NOT NULL,
    generated_text text NOT NULL,
    reference_title text NOT NULL,
    reference_text text NOT NULL,
    rouge1_f1 double precision,
    rouge2_f1 double precision,
    rougel_f1 double precision,
    bertscore_f1 double precision,
    title_similarity double precision,
    section_count_generated integer,
    section_count_reference integer,
    section_count_abs_diff integer,
    article_length_ratio double precision,
    created_at timestamp NOT NULL DEFAULT now(),
    CONSTRAINT uq_generated_articles_run UNIQUE (
        article_id,
        split,
        method,
        prompt_version,
        model_name,
        top_k
    )
);

CREATE INDEX IF NOT EXISTS ix_generated_articles_split
    ON generated_articles(split);

CREATE INDEX IF NOT EXISTS ix_generated_articles_article_id
    ON generated_articles(article_id);
