# 📄 Файл: script.py.mako
# 📂 Путь: librarian_ai/db/alembic/script.py.mako
# 📌 Назначение: Шаблон генерации миграций Alembic (макросы upgrade/downgrade)

% macro upgrade() %
    op.create_table(
        'example_table',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
% endmacro

% macro downgrade() %
    op.drop_table('example_table')
% endmacro