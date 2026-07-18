from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# Extensões que o Neo poderá ler inicialmente.
ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".js",
    ".html",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".ini",
    ".cfg",
    ".xml",
    ".tsx",
    ".jsx",
    ".ts",
}

# Pastas que não devem entrar na análise.
IGNORED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "chroma_db",
    "conversations",
    "legacy",
    "dist",
    "build",
}

# Arquivos que normalmente não agregam valor ao contexto do modelo.
IGNORED_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "package-lock.json",
    "poetry.lock",
    "uv.lock",
}

# Evita carregar arquivos enormes no prompt.
DEFAULT_MAX_FILE_SIZE = 200_000  # aproximadamente 200 KB


@dataclass
class RepositoryFile:
    """
    Representa um arquivo encontrado no repositório.
    """

    relative_path: str
    extension: str
    size_bytes: int


@dataclass
class RepositorySummary:
    """
    Resumo estrutural do repositório.
    """

    repository_name: str
    repository_path: str
    total_files: int
    total_size_bytes: int
    files_by_extension: dict[str, int]
    files: list[RepositoryFile]


def validate_repository_path(repository_path: str | Path) -> Path:
    """
    Valida e normaliza o caminho do repositório.
    """

    path = Path(repository_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"O caminho informado não existe: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"O caminho informado não é uma pasta: {path}"
        )

    return path


def is_ignored_path(path: Path, repository_root: Path) -> bool:
    """
    Verifica se alguma parte do caminho pertence à lista de pastas ignoradas.
    """

    try:
        relative_path = path.relative_to(repository_root)
    except ValueError:
        return True

    return any(
        part in IGNORED_DIRECTORIES
        for part in relative_path.parts
    )


def is_allowed_file(
    file_path: Path,
    repository_root: Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> bool:
    """
    Decide se um arquivo pode ser lido pelo Repo Reader.
    """

    if not file_path.is_file():
        return False

    if is_ignored_path(file_path, repository_root):
        return False

    if file_path.name in IGNORED_FILES:
        return False

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False

    try:
        size = file_path.stat().st_size
    except OSError:
        return False

    if size > max_file_size:
        return False

    return True


def iter_repository_files(
    repository_path: str | Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Iterable[Path]:
    """
    Percorre o repositório e retorna somente os arquivos permitidos.
    """

    repository_root = validate_repository_path(repository_path)

    for file_path in repository_root.rglob("*"):
        if is_allowed_file(
            file_path=file_path,
            repository_root=repository_root,
            max_file_size=max_file_size,
        ):
            yield file_path


def list_repo_files(
    repository_path: str | Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[RepositoryFile]:
    """
    Lista os arquivos úteis do repositório.
    """

    repository_root = validate_repository_path(repository_path)
    repository_files: list[RepositoryFile] = []

    for file_path in iter_repository_files(
        repository_root,
        max_file_size=max_file_size,
    ):
        relative_path = file_path.relative_to(repository_root)

        repository_files.append(
            RepositoryFile(
                relative_path=relative_path.as_posix(),
                extension=file_path.suffix.lower(),
                size_bytes=file_path.stat().st_size,
            )
        )

        

    return sorted(
        repository_files,
        key=lambda item: item.relative_path.lower(),
    )


def resolve_safe_file_path(
    repository_path: str | Path,
    relative_file_path: str | Path,
) -> tuple[Path, Path]:
    """
    Resolve um arquivo garantindo que ele continue dentro do repositório.

    Isso evita caminhos como:
    ../../arquivo-secreto.txt
    """

    repository_root = validate_repository_path(repository_path)

    requested_file = (
        repository_root / Path(relative_file_path)
    ).resolve()

    try:
        requested_file.relative_to(repository_root)
    except ValueError as exc:
        raise PermissionError(
            "O arquivo solicitado está fora do repositório."
        ) from exc

    return repository_root, requested_file


def read_repo_file(
    repository_path: str | Path,
    relative_file_path: str | Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> str:
    """
    Lê um arquivo específico do repositório.
    """

    repository_root, file_path = resolve_safe_file_path(
        repository_path=repository_path,
        relative_file_path=relative_file_path,
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {relative_file_path}"
        )

    if not is_allowed_file(
        file_path=file_path,
        repository_root=repository_root,
        max_file_size=max_file_size,
    ):
        raise PermissionError(
            f"O arquivo não é permitido ou ultrapassa o limite de tamanho: "
            f"{relative_file_path}"
        )

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback para alguns arquivos criados no Windows.
        return file_path.read_text(
            encoding="latin-1",
            errors="replace",
        )
    except OSError as exc:
        raise OSError(
            f"Não foi possível ler o arquivo: {relative_file_path}"
        ) from exc


def summarize_repo(
    repository_path: str | Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> RepositorySummary:
    """
    Gera um resumo estrutural do repositório.

    Neste primeiro momento, o resumo não usa LLM.
    Ele descreve objetivamente os arquivos encontrados.
    """

    repository_root = validate_repository_path(repository_path)

    files = list_repo_files(
        repository_path=repository_root,
        max_file_size=max_file_size,
    )

    files_by_extension: dict[str, int] = {}
    total_size_bytes = 0

    for file in files:
        extension = file.extension or "[sem extensão]"

        files_by_extension[extension] = (
            files_by_extension.get(extension, 0) + 1
        )

        total_size_bytes += file.size_bytes

    return RepositorySummary(
        repository_name=repository_root.name,
        repository_path=str(repository_root),
        total_files=len(files),
        total_size_bytes=total_size_bytes,
        files_by_extension=dict(
            sorted(files_by_extension.items())
        ),
        files=files,
    )


def build_repository_context(
    repository_path: str | Path,
    max_files: int = 30,
    max_characters_per_file: int = 12_000,
    max_total_characters: int = 80_000,
) -> str:
    """
    Monta um contexto textual para futuramente enviar ao modelo.

    Os limites evitam jogar o repositório inteiro no prompt.
    """

    repository_root = validate_repository_path(repository_path)
    summary = summarize_repo(repository_root)

    context_parts = [
        f"REPOSITÓRIO: {summary.repository_name}",
        f"CAMINHO: {summary.repository_path}",
        f"TOTAL DE ARQUIVOS ANALISÁVEIS: {summary.total_files}",
        "",
        "ARQUIVOS POR EXTENSÃO:",
    ]

    for extension, quantity in summary.files_by_extension.items():
        context_parts.append(f"- {extension}: {quantity}")

    context_parts.extend(
        [
            "",
            "CONTEÚDO DOS ARQUIVOS:",
        ]
    )

    total_characters = 0

    for repository_file in summary.files[:max_files]:
        if total_characters >= max_total_characters:
            context_parts.append(
                "\n[LIMITE TOTAL DE CONTEXTO ATINGIDO]"
            )
            break

        content = read_repo_file(
            repository_path=repository_root,
            relative_file_path=repository_file.relative_path,
        )

        content = content[:max_characters_per_file]

        remaining_characters = (
            max_total_characters - total_characters
        )

        content = content[:remaining_characters]

        file_section = (
            f"\n\n--- ARQUIVO: "
            f"{repository_file.relative_path} ---\n"
            f"{content}"
        )

        context_parts.append(file_section)
        total_characters += len(content)

    return "\n".join(context_parts)


def print_summary(repository_path: str | Path) -> None:
    """
    Exibe o resumo no terminal em JSON.
    """

    summary = summarize_repo(repository_path)

    print(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leitor local de repositórios do Neo."
    )

    parser.add_argument(
        "repository_path",
        nargs="?",
        default=".",
        help=(
            "Caminho do repositório. "
            "Se omitido, usa a pasta atual."
        ),
    )

    parser.add_argument(
        "--context",
        action="store_true",
        help="Exibe o contexto que seria enviado ao modelo.",
    )

    args = parser.parse_args()

    try:
        if args.context:
            print(
                build_repository_context(
                    args.repository_path
                )
            )
        else:
            print_summary(args.repository_path)

    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        OSError,
    ) as exc:
        print(f"[REPO READER] Erro: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()