#!/bin/bash

# =============================================================================
# SmartDoc Research Agent - Test Runner Script
# =============================================================================
# Este script ejecuta diferentes tipos de tests para el proyecto SmartDoc
# Soporta tests unitarios, de integración, end-to-end y de performance
# =============================================================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuración por defecto
DEFAULT_TEST_TYPE="all"
DEFAULT_VERBOSITY="normal"
DEFAULT_COVERAGE="false"
DEFAULT_PARALLEL="false"
DEFAULT_SLOW_TESTS="false"
DEFAULT_OUTPUT_FORMAT="console"

# Variables globales
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/tests"
VENV_PATH="$PROJECT_ROOT/venv"
REPORTS_DIR="$PROJECT_ROOT/test_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE} SmartDoc Research Agent - Test Suite Runner${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_section() {
    echo -e "${CYAN}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

show_help() {
    echo "SmartDoc Test Runner - Execute test suites for the research agent"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Test type: unit, integration, e2e, performance, all (default: all)"
    echo "  -v, --verbose LEVEL    Verbosity: quiet, normal, verbose, debug (default: normal)"
    echo "  -c, --coverage         Generate coverage report (default: false)"
    echo "  -p, --parallel         Run tests in parallel (default: false)"
    echo "  -s, --slow             Include slow tests (default: false)"
    echo "  -f, --format FORMAT    Output format: console, html, xml, json (default: console)"
    echo "  -o, --output DIR       Output directory for reports (default: test_reports)"
    echo "  -k, --keyword PATTERN  Run only tests matching keyword pattern"
    echo "  -m, --marker MARKER    Run only tests with specific marker"
    echo "  -x, --exitfirst        Stop on first failure"
    echo "  -l, --lf               Run last failed tests only"
    echo "  -n, --new              Run only new/modified tests"
    echo "  --setup                Setup test environment only"
    echo "  --clean                Clean test artifacts and reports"
    echo "  --env ENV              Test environment: local, docker, ci (default: local)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 -t unit -v verbose                # Run unit tests with verbose output"
    echo "  $0 -t integration -c                 # Run integration tests with coverage"
    echo "  $0 -t performance -s                 # Run performance tests including slow ones"
    echo "  $0 -k 'web_search' -v debug          # Run tests matching 'web_search' pattern"
    echo "  $0 -m 'integration' --parallel       # Run integration tests in parallel"
    echo "  $0 --setup                           # Setup test environment only"
    echo "  $0 --clean                           # Clean test artifacts"
    echo ""
}

# =============================================================================
# FUNCIONES DE CONFIGURACIÓN Y VALIDACIÓN
# =============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Verificar que estamos en el directorio correcto
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_error "requirements.txt not found. Are you in the correct directory?"
        return 1
    fi
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        return 1
    fi
    
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python version: $python_version"
    
    # Verificar virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        print_warning "Virtual environment not found at $VENV_PATH"
        print_info "Run ./scripts/setup_test_env.sh first"
        return 1
    fi
    
    # Activar virtual environment
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        print_success "Virtual environment activated"
    else
        print_error "Could not activate virtual environment"
        return 1
    fi
    
    # Verificar pytest
    if ! command -v pytest &> /dev/null; then
        print_error "pytest not found. Install test dependencies first"
        return 1
    fi
    
    local pytest_version=$(pytest --version 2>&1 | head -n1)
    print_info "$pytest_version"
    
    # Verificar directorio de tests
    if [ ! -d "$TEST_DIR" ]; then
        print_error "Tests directory not found: $TEST_DIR"
        return 1
    fi
    
    print_success "All prerequisites checked"
    return 0
}

setup_test_environment() {
    print_section "Setting Up Test Environment"
    
    # Crear directorio de reportes
    mkdir -p "$REPORTS_DIR"
    print_info "Reports directory: $REPORTS_DIR"
    
    # Crear directorio de logs de test
    mkdir -p "$PROJECT_ROOT/logs/tests"
    
    # Verificar configuración de pytest
    if [ ! -f "$PROJECT_ROOT/pytest.ini" ]; then
        print_warning "pytest.ini not found, using default configuration"
    fi
    
    # Configurar variables de entorno para tests
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export TEST_ENV="true"
    export TEST_TIMESTAMP="$TIMESTAMP"
    
    print_success "Test environment configured"
}

validate_test_structure() {
    print_section "Validating Test Structure"
    
    local required_files=(
        "$TEST_DIR/__init__.py"
        "$TEST_DIR/fixtures.py"
        "$TEST_DIR/test_data.py"
        "$TEST_DIR/unit/__init__.py"
        "$TEST_DIR/integration/__init__.py"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_warning "Missing test files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
    else
        print_success "All required test files found"
    fi
    
    # Contar archivos de test
    local unit_tests=$(find "$TEST_DIR/unit" -name "test_*.py" 2>/dev/null | wc -l)
    local integration_tests=$(find "$TEST_DIR/integration" -name "test_*.py" 2>/dev/null | wc -l)
    
    print_info "Unit tests found: $unit_tests"
    print_info "Integration tests found: $integration_tests"
}

# =============================================================================
# FUNCIONES DE EJECUCIÓN DE TESTS
# =============================================================================

build_pytest_command() {
    local test_type="$1"
    local verbosity="$2"
    local coverage="$3"
    local parallel="$4"
    local slow_tests="$5"
    local output_format="$6"
    local keyword="$7"
    local marker="$8"
    local exitfirst="$9"
    local lastfailed="${10}"
    
    local cmd="pytest"
    
    # Configurar directorio de tests según tipo
    case "$test_type" in
        "unit")
            cmd="$cmd $TEST_DIR/unit/"
            ;;
        "integration")
            cmd="$cmd $TEST_DIR/integration/"
            ;;
        "e2e")
            cmd="$cmd $TEST_DIR/integration/test_full_workflow.py"
            ;;
        "performance")
            cmd="$cmd -m performance"
            ;;
        "all")
            cmd="$cmd $TEST_DIR/"
            ;;
        *)
            cmd="$cmd $TEST_DIR/"
            ;;
    esac
    
    # Configurar verbosidad
    case "$verbosity" in
        "quiet")
            cmd="$cmd -q"
            ;;
        "verbose")
            cmd="$cmd -v"
            ;;
        "debug")
            cmd="$cmd -vv -s"
            ;;
        *)
            cmd="$cmd -v"
            ;;
    esac
    
    # Agregar coverage si está habilitado
    if [ "$coverage" = "true" ]; then
        cmd="$cmd --cov=app --cov-report=html:$REPORTS_DIR/coverage_html_$TIMESTAMP"
        cmd="$cmd --cov-report=xml:$REPORTS_DIR/coverage_$TIMESTAMP.xml"
        cmd="$cmd --cov-report=term"
    fi
    
    # Configurar paralelización
    if [ "$parallel" = "true" ]; then
        local cpu_count=$(nproc 2>/dev/null || echo "2")
        cmd="$cmd -n $cpu_count"
    fi
    
    # Incluir/excluir tests lentos
    if [ "$slow_tests" = "false" ]; then
        cmd="$cmd -m 'not slow'"
    fi
    
    # Configurar formato de output
    case "$output_format" in
        "html")
            cmd="$cmd --html=$REPORTS_DIR/report_$TIMESTAMP.html --self-contained-html"
            ;;
        "xml")
            cmd="$cmd --junitxml=$REPORTS_DIR/results_$TIMESTAMP.xml"
            ;;
        "json")
            cmd="$cmd --json-report --json-report-file=$REPORTS_DIR/results_$TIMESTAMP.json"
            ;;
    esac
    
    # Filtros adicionales
    if [ -n "$keyword" ]; then
        cmd="$cmd -k '$keyword'"
    fi
    
    if [ -n "$marker" ]; then
        cmd="$cmd -m '$marker'"
    fi
    
    # Opciones de ejecución
    if [ "$exitfirst" = "true" ]; then
        cmd="$cmd -x"
    fi
    
    if [ "$lastfailed" = "true" ]; then
        cmd="$cmd --lf"
    fi
    
    # Configuraciones adicionales
    cmd="$cmd --tb=short"
    cmd="$cmd --strict-markers"
    cmd="$cmd --disable-warnings"
    
    echo "$cmd"
}

run_unit_tests() {
    print_section "Running Unit Tests"
    
    local cmd=$(build_pytest_command "unit" "$VERBOSITY" "$COVERAGE" "$PARALLEL" "$SLOW_TESTS" "$OUTPUT_FORMAT" "$KEYWORD" "$MARKER" "$EXITFIRST" "$LASTFAILED")
    
    print_info "Command: $cmd"
    echo ""
    
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

run_integration_tests() {
    print_section "Running Integration Tests"
    
    local cmd=$(build_pytest_command "integration" "$VERBOSITY" "$COVERAGE" "$PARALLEL" "$SLOW_TESTS" "$OUTPUT_FORMAT" "$KEYWORD" "$MARKER" "$EXITFIRST" "$LASTFAILED")
    
    print_info "Command: $cmd"
    echo ""
    
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

run_e2e_tests() {
    print_section "Running End-to-End Tests"
    
    local cmd=$(build_pytest_command "e2e" "$VERBOSITY" "$COVERAGE" "$PARALLEL" "$SLOW_TESTS" "$OUTPUT_FORMAT" "$KEYWORD" "$MARKER" "$EXITFIRST" "$LASTFAILED")
    
    print_info "Command: $cmd"
    echo ""
    
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "End-to-end tests passed"
    else
        print_error "End-to-end tests failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

run_performance_tests() {
    print_section "Running Performance Tests"
    
    print_warning "Performance tests may take longer to complete"
    
    local cmd=$(build_pytest_command "performance" "$VERBOSITY" "false" "false" "true" "$OUTPUT_FORMAT" "$KEYWORD" "$MARKER" "$EXITFIRST" "$LASTFAILED")
    
    print_info "Command: $cmd"
    echo ""
    
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Performance tests passed"
    else
        print_error "Performance tests failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

run_all_tests() {
    print_section "Running All Tests"
    
    local overall_exit_code=0
    
    # Unit tests
    run_unit_tests
    if [ $? -ne 0 ]; then
        overall_exit_code=1
    fi
    
    echo ""
    
    # Integration tests
    run_integration_tests
    if [ $? -ne 0 ]; then
        overall_exit_code=1
    fi
    
    echo ""
    
    # Performance tests (solo si se solicitan explícitamente)
    if [ "$SLOW_TESTS" = "true" ]; then
        run_performance_tests
        if [ $? -ne 0 ]; then
            overall_exit_code=1
        fi
    fi
    
    return $overall_exit_code
}

# =============================================================================
# FUNCIONES DE UTILIDAD Y REPORTES
# =============================================================================

generate_test_summary() {
    print_section "Test Summary"
    
    local reports_found=$(find "$REPORTS_DIR" -name "*$TIMESTAMP*" 2>/dev/null | wc -l)
    
    if [ $reports_found -gt 0 ]; then
        print_info "Generated reports:"
        find "$REPORTS_DIR" -name "*$TIMESTAMP*" -type f | while read -r file; do
            echo "  - $(basename "$file")"
        done
    fi
    
    # Mostrar coverage summary si existe
    if [ -f "$REPORTS_DIR/coverage_$TIMESTAMP.xml" ]; then
        print_info "Coverage report generated: coverage_$TIMESTAMP.xml"
    fi
    
    # Mostrar logs si existen
    local log_files=$(find "$PROJECT_ROOT/logs/tests" -name "*$TIMESTAMP*" 2>/dev/null | wc -l)
    if [ $log_files -gt 0 ]; then
        print_info "Test logs available in logs/tests/"
    fi
}

clean_test_artifacts() {
    print_section "Cleaning Test Artifacts"
    
    # Limpiar cache de pytest
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    
    # Limpiar reportes antiguos (más de 7 días)
    if [ -d "$REPORTS_DIR" ]; then
        find "$REPORTS_DIR" -type f -mtime +7 -delete 2>/dev/null || true
        print_success "Old test reports cleaned"
    fi
    
    # Limpiar logs antiguos
    if [ -d "$PROJECT_ROOT/logs/tests" ]; then
        find "$PROJECT_ROOT/logs/tests" -type f -mtime +7 -delete 2>/dev/null || true
        print_success "Old test logs cleaned"
    fi
    
    # Limpiar archivos temporales
    find "$PROJECT_ROOT" -name "*.tmp" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".coverage*" -delete 2>/dev/null || true
    
    print_success "Test artifacts cleaned"
}

run_test_discovery() {
    print_section "Test Discovery"
    
    print_info "Discovering available tests..."
    
    pytest --collect-only -q "$TEST_DIR" 2>/dev/null | grep -E "<Module|<Function" | while read -r line; do
        echo "  $line"
    done
    
    echo ""
    print_info "Available test markers:"
    pytest --markers 2>/dev/null | grep -E "^@pytest.mark" | head -10
}

check_test_environment_health() {
    print_section "Test Environment Health Check"
    
    # Verificar imports críticos
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from app.agents.core.smart_agent import SmartDocAgent
    print('✅ SmartDocAgent import successful')
except Exception as e:
    print(f'❌ SmartDocAgent import failed: {e}')

try:
    from app.agents.tools.web.web_search_tool import WebSearchTool
    print('✅ WebSearchTool import successful')
except Exception as e:
    print(f'❌ WebSearchTool import failed: {e}')

try:
    import pytest
    print(f'✅ pytest {pytest.__version__} available')
except Exception as e:
    print(f'❌ pytest not available: {e}')

try:
    from tests.fixtures import MOCK_SEARCH_RESULTS
    print('✅ Test fixtures loaded successfully')
except Exception as e:
    print(f'❌ Test fixtures failed: {e}')
"
}

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

main() {
    # Variables por defecto
    TEST_TYPE="$DEFAULT_TEST_TYPE"
    VERBOSITY="$DEFAULT_VERBOSITY"
    COVERAGE="$DEFAULT_COVERAGE"
    PARALLEL="$DEFAULT_PARALLEL"
    SLOW_TESTS="$DEFAULT_SLOW_TESTS"
    OUTPUT_FORMAT="$DEFAULT_OUTPUT_FORMAT"
    OUTPUT_DIR="$REPORTS_DIR"
    KEYWORD=""
    MARKER=""
    EXITFIRST="false"
    LASTFAILED="false"
    NEW_ONLY="false"
    SETUP_ONLY="false"
    CLEAN_ONLY="false"
    TEST_ENV="local"
    
    # Parsear argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSITY="$2"
                shift 2
                ;;
            -c|--coverage)
                COVERAGE="true"
                shift
                ;;
            -p|--parallel)
                PARALLEL="true"
                shift
                ;;
            -s|--slow)
                SLOW_TESTS="true"
                shift
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -k|--keyword)
                KEYWORD="$2"
                shift 2
                ;;
            -m|--marker)
                MARKER="$2"
                shift 2
                ;;
            -x|--exitfirst)
                EXITFIRST="true"
                shift
                ;;
            -l|--lf)
                LASTFAILED="true"
                shift
                ;;
            -n|--new)
                NEW_ONLY="true"
                shift
                ;;
            --setup)
                SETUP_ONLY="true"
                shift
                ;;
            --clean)
                CLEAN_ONLY="true"
                shift
                ;;
            --env)
                TEST_ENV="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_header
    
    # Manejo de opciones especiales
    if [ "$CLEAN_ONLY" = "true" ]; then
        clean_test_artifacts
        exit 0
    fi
    
    if [ "$SETUP_ONLY" = "true" ]; then
        check_prerequisites
        setup_test_environment
        validate_test_structure
        check_test_environment_health
        exit 0
    fi
    
    # Verificar prerequisites
    if ! check_prerequisites; then
        print_error "Prerequisites check failed"
        exit 1
    fi
    
    # Setup del entorno
    setup_test_environment
    validate_test_structure
    
    # Actualizar REPORTS_DIR si se especificó uno custom
    if [ "$OUTPUT_DIR" != "$REPORTS_DIR" ]; then
        REPORTS_DIR="$OUTPUT_DIR"
        mkdir -p "$REPORTS_DIR"
    fi
    
    # Ejecutar tests según el tipo solicitado
    local exit_code=0
    
    case "$TEST_TYPE" in
        "unit")
            run_unit_tests
            exit_code=$?
            ;;
        "integration")
            run_integration_tests
            exit_code=$?
            ;;
        "e2e")
            run_e2e_tests
            exit_code=$?
            ;;
        "performance")
            run_performance_tests
            exit_code=$?
            ;;
        "all")
            run_all_tests
            exit_code=$?
            ;;
        "discover")
            run_test_discovery
            exit_code=0
            ;;
        "health")
            check_test_environment_health
            exit_code=0
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            print_info "Valid types: unit, integration, e2e, performance, all, discover, health"
            exit 1
            ;;
    esac
    
    echo ""
    generate_test_summary
    
    # Mensaje final
    echo ""
    if [ $exit_code -eq 0 ]; then
        print_success "All tests completed successfully! 🎉"
    else
        print_error "Some tests failed. Check the output above for details."
        print_info "You can run with -v verbose or --lf to rerun only failed tests"
    fi
    
    echo ""
    print_info "Test reports saved to: $REPORTS_DIR"
    print_info "Run '$0 --help' for more options"
    
    exit $exit_code
}

# =============================================================================
# EJECUCIÓN
# =============================================================================

# Verificar que el script se ejecute desde el directorio correcto
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: This script must be run from the agent-api directory${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    echo -e "${YELLOW}Expected to find: requirements.txt${NC}"
    exit 1
fi

# Ejecutar función principal con todos los argumentos
main "$@"