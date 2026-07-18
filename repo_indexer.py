from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import chromadb

from repo_reader import (
    list_repo_files,
    read_repo_file,
    validate_repository_path,
)
from vector_memory import generate_embedding


CHROMA_DB_PATH = "./chroma_db"
REPOSITORY_COLLECTION_NAME = "neo_repository"

DEFAULT_CHUNK_SIZE_LINES = 120
DEFAULT_CHUNK_OVERLAP_LINES = 20
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_PREVIEW_CHARACTERS = 1800


client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

repository_collection = client.get_or_create_collection(
    name=REPOSITORY_COLLECTION_NAME
)


def log(message: str) -> None:
    """
    Exibe uma mensagem padronizada do Repo Indexer.
    """

    print(f"[REPO INDEXER] {message}")


def normalize_repository_path(
    repository_path: str | Path,
) -> str:
    """
    Retorna uma representação estável do caminho do repositório.
    """

    repository_root = validate_repository_path(
        repository_path
    )

    return str(repository_root)


def create_chunk_id(
    repository_path: str,
    relative_path: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Cria um identificador determinístico para cada chunk.

    O conteúdo entra no hash para que alterações no arquivo
    também alterem seu identificador.
    """

    raw_id = (
        f"{repository_path}|"
        f"{relative_path}|"
        f"{chunk_index}|"
        f"{content}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def split_content_into_chunks(
    content: str,
    chunk_size_lines: int = DEFAULT_CHUNK_SIZE_LINES,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[dict[str, Any]]:
    """
    Divide o conteúdo em blocos de linhas com sobreposição.

    A sobreposição evita perder contexto quando uma função
    começa no final de um chunk e termina no próximo.
    """

    if chunk_size_lines <= 0:
        raise ValueError(
            "chunk_size_lines deve ser maior que zero."
        )

    if overlap_lines < 0:
        raise ValueError(
            "overlap_lines não pode ser negativo."
        )

    if overlap_lines >= chunk_size_lines:
        raise ValueError(
            "overlap_lines deve ser menor que chunk_size_lines."
        )

    lines = content.splitlines()

    if not lines:
        return []

    chunks: list[dict[str, Any]] = []

    start_index = 0
    chunk_index = 0
    step = chunk_size_lines - overlap_lines

    while start_index < len(lines):
        end_index = min(
            start_index + chunk_size_lines,
            len(lines),
        )

        chunk_lines = lines[start_index:end_index]
        chunk_content = "\n".join(
            chunk_lines
        ).strip()

        if chunk_content:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_line": start_index + 1,
                    "end_line": end_index,
                    "content": chunk_content,
                }
            )

        chunk_index += 1
        start_index += step

    return chunks


def clear_repository_index(
    repository_path: str | Path,
) -> None:
    """
    Remove do ChromaDB os chunks antigos desse repositório.

    Isso evita manter arquivos apagados ou trechos
    desatualizados.
    """

    normalized_path = normalize_repository_path(
        repository_path
    )

    repository_collection.delete(
        where={
            "repository_path": normalized_path
        }
    )


def build_chunk_document(
    repository_name: str,
    relative_path: str,
    extension: str,
    start_line: int,
    end_line: int,
    content: str,
) -> str:
    """
    Monta o texto armazenado e vetorizado.

    Os metadados textuais ajudam o mecanismo de embeddings
    a compreender que o conteúdo pertence a um repositório
    e a um arquivo de código específico.
    """

    return (
        f"REPOSITÓRIO: {repository_name}\n"
        f"ARQUIVO: {relative_path}\n"
        f"EXTENSÃO: {extension}\n"
        f"LINHAS: {start_line}-{end_line}\n\n"
        f"CONTEÚDO:\n"
        f"{content}"
    )


def index_chunk(
    repository_name: str,
    repository_path: str,
    relative_path: str,
    extension: str,
    chunk: dict[str, Any],
) -> None:
    """
    Gera o embedding e salva um chunk no ChromaDB.
    """

    document = build_chunk_document(
        repository_name=repository_name,
        relative_path=relative_path,
        extension=extension,
        start_line=chunk["start_line"],
        end_line=chunk["end_line"],
        content=chunk["content"],
    )

    embedding = generate_embedding(
        document
    )

    chunk_id = create_chunk_id(
        repository_path=repository_path,
        relative_path=relative_path,
        chunk_index=chunk["chunk_index"],
        content=chunk["content"],
    )

    repository_collection.upsert(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[
            {
                "repository_name": repository_name,
                "repository_path": repository_path,
                "relative_path": relative_path,
                "extension": extension,
                "chunk_index": chunk["chunk_index"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
        ],
    )


def index_repository(
    repository_path: str | Path,
    chunk_size_lines: int = DEFAULT_CHUNK_SIZE_LINES,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> dict[str, Any]:
    """
    Lê os arquivos permitidos, divide em chunks
    e salva os chunks no ChromaDB.
    """

    repository_root = validate_repository_path(
        repository_path
    )

    normalized_path = str(repository_root)

    repository_files = list_repo_files(
        repository_root
    )

    clear_repository_index(
        repository_root
    )

    indexed_files = 0
    indexed_chunks = 0
    skipped_files: list[str] = []

    for repository_file in repository_files:
        try:
            content = read_repo_file(
                repository_path=repository_root,
                relative_file_path=(
                    repository_file.relative_path
                ),
            )

        except (
            FileNotFoundError,
            PermissionError,
            OSError,
        ) as exc:
            log(
                "Arquivo ignorado: "
                f"{repository_file.relative_path} — {exc}"
            )

            skipped_files.append(
                repository_file.relative_path
            )

            continue

        chunks = split_content_into_chunks(
            content=content,
            chunk_size_lines=chunk_size_lines,
            overlap_lines=overlap_lines,
        )

        if not chunks:
            continue

        indexed_files += 1

        for chunk in chunks:
            try:
                index_chunk(
                    repository_name=repository_root.name,
                    repository_path=normalized_path,
                    relative_path=(
                        repository_file.relative_path
                    ),
                    extension=repository_file.extension,
                    chunk=chunk,
                )

            except Exception as exc:
                log(
                    "Falha ao indexar "
                    f"{repository_file.relative_path} "
                    f"linhas {chunk['start_line']}-"
                    f"{chunk['end_line']}: {exc}"
                )

                continue

            indexed_chunks += 1

            log(
                f"{repository_file.relative_path} "
                f"linhas {chunk['start_line']}-"
                f"{chunk['end_line']}"
            )

    return {
        "repository_name": repository_root.name,
        "repository_path": normalized_path,
        "files_found": len(repository_files),
        "files_indexed": indexed_files,
        "chunks_indexed": indexed_chunks,
        "skipped_files": skipped_files,
    }


def normalize_search_text(
    text: str,
) -> str:
    """
    Normaliza texto para comparação literal:

    - remove acentos;
    - converte para minúsculas;
    - separa camelCase;
    - transforma símbolos em espaços.
    """

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    normalized = unicodedata.normalize(
        "NFKD",
        text,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    normalized = normalized.lower()

    return re.sub(
        r"[^a-z0-9_]+",
        " ",
        normalized,
    ).strip()


def extract_search_terms(
    query: str,
) -> list[str]:
    """
    Extrai termos úteis e cria variações comuns
    encontradas em código-fonte.

    Também traduz alguns conceitos de português
    para os nomes normalmente usados no código.
    """

    ignored_terms = {
        "a",
        "as",
        "como",
        "configurado",
        "configurada",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "esta",
        "foi",
        "funciona",
        "implementado",
        "implementada",
        "onde",
        "o",
        "os",
        "qual",
        "que",
        "um",
        "uma",
    }

    normalized_query = normalize_search_text(
        query
    )

    terms = [
        term
        for term in normalized_query.split()
        if (
            len(term) >= 3
            and term not in ignored_terms
        )
    ]

    expanded_terms = set(terms)

    if len(terms) >= 2:
        expanded_terms.update(
            {
                "_".join(terms),
                "".join(terms),
                " ".join(terms),
            }
        )

    query_text = " ".join(terms)

    concept_aliases = {
        "deep mode": {
            "deep_mode",
            "deepmode",
            "deep model",
            "deep_model",
            "toggle deep mode",
            "toggledeepmode",
        },
        "memoria vetorial": {
            "vector memory",
            "vector_memory",
            "search memory",
            "search_memory",
            "save memory",
            "save_memory",
            "embedding",
            "embeddings",
            "chromadb",
            "chroma db",
            "rag",
        },
        "reconhecimento voz": {
            "speech recognition",
            "speechrecognition",
            "setup speech recognition",
            "setupspeechrecognition",
            "webkit speech recognition",
            "webkitspeechrecognition",
            "start recognition",
            "startrecognition",
            "recognition",
            "voice",
            "voice button",
            "voicebtn",
        },
    }

    for concept, aliases in concept_aliases.items():
        concept_words = concept.split()

        if all(word in query_text for word in concept_words):
            expanded_terms.update(aliases)

    return sorted(expanded_terms)


def count_term_occurrences(
    normalized_text: str,
    normalized_term: str,
) -> int:
    """
    Conta correspondências completas de um termo.

    Isso reduz falsos positivos em que um termo curto
    aparece dentro de outra palavra.
    """

    if not normalized_term:
        return 0

    pattern = (
        r"(?<![a-z0-9_])"
        + re.escape(normalized_term)
        + r"(?![a-z0-9_])"
    )

    return len(
        re.findall(
            pattern,
            normalized_text,
        )
    )


def calculate_literal_score(
    document: str,
    relative_path: str,
    search_terms: list[str],
) -> int:
    """
    Calcula a pontuação literal de um resultado.

    Correspondências no caminho do arquivo recebem
    mais peso que correspondências no conteúdo.
    """

    normalized_document = normalize_search_text(
        document
    )

    normalized_file_path = normalize_search_text(
        relative_path
    )

    literal_score = 0

    for term in search_terms:
        normalized_term = normalize_search_text(
            term
        )

        document_occurrences = count_term_occurrences(
            normalized_document,
            normalized_term,
        )

        path_occurrences = count_term_occurrences(
            normalized_file_path,
            normalized_term,
        )

        literal_score += document_occurrences * 18
        literal_score += path_occurrences * 60

    implementation_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
    }

    file_extension = Path(
        relative_path
    ).suffix.lower()

    if file_extension in implementation_extensions:
        literal_score += 25

    if relative_path.lower() == "readme.md":
        literal_score -= 30

    return literal_score


def retrieve_repository_candidates(
    query: str,
    repository_path: str | Path,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """
    Recupera candidatos semanticamente relevantes
    diretamente do ChromaDB.
    """

    repository_root = validate_repository_path(
        repository_path
    )

    normalized_path = str(repository_root)

    total_chunks = repository_collection.count()

    if total_chunks == 0:
        return []

    query_embedding = generate_embedding(
        query
    )

    safe_candidate_limit = min(
        candidate_limit,
        total_chunks,
    )

    results = repository_collection.query(
        query_embeddings=[query_embedding],
        n_results=safe_candidate_limit,
        where={
            "repository_path": normalized_path
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    candidates: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if not document or not metadata:
            continue

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    return candidates


def rank_repository_candidates(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reordena os candidatos combinando:

    - distância vetorial;
    - correspondência literal;
    - termos e símbolos encontrados no código.
    """

    search_terms = extract_search_terms(
        query
    )

    ranked_results: list[dict[str, Any]] = []

    for candidate in candidates:
        document = candidate["document"]
        metadata = candidate["metadata"]
        distance = candidate["distance"]

        relative_path = metadata.get(
            "relative_path",
            "",
        )

        literal_score = calculate_literal_score(
            document=document,
            relative_path=relative_path,
            search_terms=search_terms,
        )

        # Quanto menor o resultado final, melhor.
        hybrid_score = (
            distance - literal_score
        )

        ranked_results.append(
            {
                "relative_path": relative_path,
                "start_line": metadata.get(
                    "start_line"
                ),
                "end_line": metadata.get(
                    "end_line"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "extension": metadata.get(
                    "extension"
                ),
                "distance": distance,
                "literal_score": literal_score,
                "hybrid_score": hybrid_score,
                "content": document,
            }
        )

    ranked_results.sort(
        key=lambda result: result["hybrid_score"]
    )

    return ranked_results


def search_repository(
    query: str,
    repository_path: str | Path,
    limit: int = DEFAULT_SEARCH_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """
    Realiza busca híbrida no repositório:

    1. Recupera candidatos por embeddings.
    2. Reordena por correspondência literal.
    3. Retorna somente os melhores resultados.
    """

    if not query.strip():
        raise ValueError(
            "A consulta não pode estar vazia."
        )

    if limit <= 0:
        raise ValueError(
            "O limite deve ser maior que zero."
        )

    candidates = retrieve_repository_candidates(
        query=query,
        repository_path=repository_path,
        candidate_limit=max(
            limit,
            candidate_limit,
        ),
    )

    ranked_results = rank_repository_candidates(
        query=query,
        candidates=candidates,
    )

    return ranked_results[:limit]


def print_search_results(
    query: str,
    repository_path: str | Path,
    limit: int = DEFAULT_SEARCH_LIMIT,
    preview_characters: int = DEFAULT_PREVIEW_CHARACTERS,
) -> None:
    """
    Exibe no terminal os chunks encontrados.
    """

    results = search_repository(
        query=query,
        repository_path=repository_path,
        limit=limit,
    )

    if not results:
        log(
            "Nenhum resultado encontrado. "
            "Verifique se o repositório foi indexado."
        )
        return

    print(
        f"\n[REPO INDEXER] "
        f"Resultados para: {query}\n"
    )

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 70)

        print(
            f"RESULTADO {position}\n"
            f"Arquivo: {result['relative_path']}\n"
            f"Linhas: {result['start_line']}-"
            f"{result['end_line']}\n"
            f"Distância vetorial: "
            f"{result['distance']:.4f}\n"
            f"Pontuação literal: "
            f"{result['literal_score']}\n"
            f"Pontuação híbrida: "
            f"{result['hybrid_score']:.4f}\n"
        )

        content = result["content"]

        content_preview = content[
            :preview_characters
        ]

        print(content_preview)

        if len(content) > preview_characters:
            print(
                "\n[CONTEÚDO CORTADO NO TERMINAL]"
            )

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Indexador vetorial de repositórios do Neo."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Indexa um repositório local.",
    )

    index_parser.add_argument(
        "repository_path",
        nargs="?",
        default=".",
        help="Caminho do repositório.",
    )

    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_LINES,
        help="Quantidade de linhas por chunk.",
    )

    index_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_LINES,
        help="Quantidade de linhas sobrepostas.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Busca informações no repositório indexado.",
    )

    search_parser.add_argument(
        "repository_path",
        nargs="?",
        default=".",
        help="Caminho do repositório.",
    )

    search_parser.add_argument(
        "query",
        help="Pergunta ou termo para buscar.",
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Quantidade máxima de resultados.",
    )

    args = parser.parse_args()

    try:
        if args.command == "index":
            summary = index_repository(
                repository_path=args.repository_path,
                chunk_size_lines=args.chunk_size,
                overlap_lines=args.overlap,
            )

            print(
                "\n[REPO INDEXER] "
                "Indexação concluída:\n"
            )

            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        elif args.command == "search":
            print_search_results(
                query=args.query,
                repository_path=args.repository_path,
                limit=args.limit,
            )

    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        ValueError,
        OSError,
    ) as exc:
        log(
            f"Erro: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()