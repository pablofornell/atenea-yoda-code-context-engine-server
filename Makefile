setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .
	docker compose up -d
	ollama pull nomic-embed-text

run:
	. .venv/bin/activate && atenea-server

clean-index:
	@for col in $$(curl -s http://localhost:6333/collections | python3 -c "import sys,json; [print(c['name']) for c in json.load(sys.stdin)['result']['collections']]"); do \
		echo "Deleting collection: $$col"; \
		curl -s -X DELETE http://localhost:6333/collections/$$col; \
		echo; \
	done

clean:
	docker compose down
	rm -rf .venv
