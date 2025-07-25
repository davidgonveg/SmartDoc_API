"""
SmartDoc Agent - ReAct Pattern Templates
Templates de prompts para el agente de investigación usando patrón ReAct
"""

from langchain.prompts import PromptTemplate
from typing import Dict, Any

# System prompt principal para el agente
SMARTDOC_SYSTEM_PROMPT = """Eres SmartDoc, un agente de investigación inteligente especializado en investigar cualquier tema usando múltiples fuentes y herramientas.

Tu personalidad:
- Meticuloso y analítico en tu investigación
- Siempre buscas múltiples fuentes para validar información
- Explicas tu proceso de pensamiento paso a paso
- Eres honesto sobre limitaciones y nivel de confianza
- Priorizas calidad sobre velocidad

Tu proceso de trabajo:
1. Entiendes la pregunta/tema a investigar
2. Planificas qué herramientas necesitas usar
3. Ejecutas búsquedas y análisis sistemáticamente
4. Validas información cruzando fuentes
5. Sintetizas hallazgos en respuesta coherente
6. Proporcionas fuentes y nivel de confianza

Siempre usa el formato ReAct que se te especifica."""

# Template principal ReAct
REACT_MAIN_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
- Always use tools when you need current information
- Explain your reasoning step by step
- Provide sources when possible
- Be thorough in your research

Session Topic: {topic}
Session Objectives: {objectives}

Previous Conversation:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# Template para investigación académica
ACADEMIC_RESEARCH_TEMPLATE = """Eres SmartDoc configurado en modo ACADÉMICO. 

Enfoque en:
- Fuentes científicas y académicas confiables
- Evidencia basada en datos y estudios
- Citas y referencias precisas
- Análisis crítico de metodologías
- Identificación de controversias o debates

{base_template}

Recuerda: Prioriza rigor científico y evidencia empírica."""

# Template para investigación empresarial
BUSINESS_RESEARCH_TEMPLATE = """Eres SmartDoc configurado en modo EMPRESARIAL.

Enfoque en:
- Métricas de negocio y KPIs relevantes
- Tendencias de mercado y competencia
- ROI y análisis costo-beneficio
- Casos de estudio empresariales
- Aplicabilidad práctica inmediata

{base_template}

Recuerda: Enfócate en impacto empresarial y decisiones ejecutivas."""

# Template para investigación técnica
TECHNICAL_RESEARCH_TEMPLATE = """Eres SmartDoc configurado en modo TÉCNICO.

Enfoque en:
- Especificaciones técnicas detalladas
- Implementaciones y arquitecturas
- Comparativas de rendimiento
- Documentación oficial y APIs
- Best practices y consideraciones técnicas

{base_template}

Recuerda: Prioriza precisión técnica y detalles de implementación."""

# Template para síntesis final
SYNTHESIS_TEMPLATE = """Basándote en toda la información recopilada durante esta investigación, proporciona una síntesis final que incluya:

INFORMACIÓN RECOPILADA:
{gathered_info}

FUENTES CONSULTADAS:
{sources_used}

Estructura tu respuesta final así:

## Resumen Ejecutivo
[Síntesis de 2-3 párrafos de los hallazgos principales]

## Hallazgos Detallados
[Análisis profundo organizado por temas/categorías]

## Fuentes y Referencias
[Lista de todas las fuentes consultadas con relevancia]

## Nivel de Confianza
[Alto/Medio/Bajo] - [Explicación del nivel de confianza]

## Limitaciones y Consideraciones
[Qué información podría faltar o necesitar verificación adicional]

## Recomendaciones
[Si aplica, próximos pasos o acciones recomendadas]"""

# Template para validación de fuentes
SOURCE_VALIDATION_TEMPLATE = """Evalúa la credibilidad y relevancia de esta fuente:

FUENTE: {source_info}
CONTENIDO: {content_summary}
CONTEXTO: {research_context}

Evalúa en una escala de 1-10:
- Credibilidad de la fuente: [puntuación]
- Relevancia al tema: [puntuación]  
- Actualidad de la información: [puntuación]
- Rigor metodológico (si aplica): [puntuación]

Justificación breve: [explicación de las puntuaciones]
Recomendación: [usar/descartar/usar con precaución]"""

# Template para detección de conflictos
CONFLICT_DETECTION_TEMPLATE = """Analiza posibles conflictos o inconsistencias entre estas fuentes:

FUENTE A: {source_a}
INFORMACIÓN A: {info_a}

FUENTE B: {source_b} 
INFORMACIÓN B: {info_b}

¿Hay conflictos? [Sí/No]
Si hay conflictos:
- Describe la discrepancia específica
- Evalúa cuál fuente es más confiable y por qué
- Sugiere cómo resolver o presentar el conflicto

Si no hay conflictos:
- Confirma que la información es consistente
- Identifica cómo se complementan las fuentes"""

# Template para generación de reportes
REPORT_GENERATION_TEMPLATE = """Genera un reporte {report_style} sobre: {topic}

INFORMACIÓN RECOPILADA:
{research_data}

ESTILO REQUERIDO: {report_style}
- Ejecutivo: Conciso, enfocado en decisiones, métricas clave
- Académico: Riguroso, con citas, metodología clara
- Técnico: Detalles de implementación, especificaciones

SECCIONES A INCLUIR: {sections}

Formato de salida: {output_format}

Genera el reporte completo siguiendo las mejores prácticas para el estilo seleccionado."""

class ReactTemplates:
    """Clase para gestionar templates de ReAct"""
    
    @staticmethod
    def get_main_template() -> PromptTemplate:
        """Template principal ReAct"""
        return PromptTemplate(
            template=REACT_MAIN_TEMPLATE,
            input_variables=["tools", "topic", "objectives", "chat_history", "input", "agent_scratchpad"]
        )
    
    @staticmethod
    def get_research_template(style: str = "general") -> PromptTemplate:
        """Template según estilo de investigación"""
        
        base = REACT_MAIN_TEMPLATE
        
        templates = {
            "academic": ACADEMIC_RESEARCH_TEMPLATE.format(base_template=base),
            "business": BUSINESS_RESEARCH_TEMPLATE.format(base_template=base),
            "technical": TECHNICAL_RESEARCH_TEMPLATE.format(base_template=base),
            "general": base
        }
        
        template = templates.get(style, base)
        
        return PromptTemplate(
            template=template,
            input_variables=["tools", "topic", "objectives", "chat_history", "input", "agent_scratchpad"]
        )
    
    @staticmethod
    def get_synthesis_template() -> PromptTemplate:
        """Template para síntesis final"""
        return PromptTemplate(
            template=SYNTHESIS_TEMPLATE,
            input_variables=["gathered_info", "sources_used"]
        )
    
    @staticmethod
    def get_source_validation_template() -> PromptTemplate:
        """Template para validar fuentes"""
        return PromptTemplate(
            template=SOURCE_VALIDATION_TEMPLATE,
            input_variables=["source_info", "content_summary", "research_context"]
        )
    
    @staticmethod
    def get_conflict_detection_template() -> PromptTemplate:
        """Template para detectar conflictos entre fuentes"""
        return PromptTemplate(
            template=CONFLICT_DETECTION_TEMPLATE,
            input_variables=["source_a", "info_a", "source_b", "info_b"]
        )
    
    @staticmethod
    def get_report_template() -> PromptTemplate:
        """Template para generación de reportes"""
        return PromptTemplate(
            template=REPORT_GENERATION_TEMPLATE,
            input_variables=["topic", "research_data", "report_style", "sections", "output_format"]
        )

# Configuraciones de prompt por defecto
DEFAULT_PROMPT_CONFIGS = {
    "temperature": 0.1,  # Baja para consistencia en research
    "max_tokens": 4096,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# Ejemplos de uso para testing
EXAMPLE_INPUTS = {
    "simple_query": {
        "topic": "Inteligencia Artificial",
        "objectives": ["Aplicaciones actuales", "Tendencias futuras"],
        "input": "¿Cuáles son las aplicaciones más prometedoras de IA en 2024?"
    },
    "complex_query": {
        "topic": "Cambio Climático",
        "objectives": ["Impacto económico", "Soluciones tecnológicas", "Políticas gubernamentales"],
        "input": "Analiza el impacto económico del cambio climático y las principales estrategias de mitigación"
    }
}