"""Learning System Module"""

from typing import Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .config.config_loader import ConfigLoader, LearningConfig
from .storage.memory_manager import MemoryManager
from .knowledge.knowledge_graph import KnowledgeGraph
from .evolution.knowledge_evolution import KnowledgeEvolution

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"

class LearningMode(Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"

@dataclass
class LearningState:
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    knowledge_state: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class LearningSystem:
    def __init__(self, config_path: Optional[str] = None, 
                 storage_path: Optional[str] = None,
                 graph_path: Optional[str] = None):
        self.config_loader = ConfigLoader(config_path)
        self.config: LearningConfig = self.config_loader.load_config()
        self.memory_manager = MemoryManager(storage_path)
        self.knowledge_graph = KnowledgeGraph(graph_path)
        self.knowledge_evolution = KnowledgeEvolution(self.knowledge_graph)
        self.learning_state = LearningState()
        self.initialize_system()

    def initialize_system(self):
        """Initialize the learning system state and configuration"""
        self.config_loader.validate_config()
        
        # Initialize performance metrics
        self.learning_state.performance_metrics = {
            "accuracy": 0.0,
            "efficiency": 0.0,
            "adaptability": 0.0
        }
        
        # Restore knowledge state from storage
        for category in ["concepts", "rules", "skills", "experiences"]:
            stored_state = self.memory_manager.get_knowledge_state(category)
            if stored_state:
                self.learning_state.knowledge_state[category] = stored_state['state']
            else:
                self.learning_state.knowledge_state[category] = {}
        
        # Initialize resource usage state
        self.learning_state.resource_usage = {
            "memory": 0.0,
            "cpu": 0.0,
            "storage": 0.0
        }

    def learn_from_content(self, content: Any, content_type: ContentType, 
                          mode: LearningMode, retention_period: Optional[timedelta] = None) -> bool:
        """Learn from given content"""
        try:
            # Validate content type and learning mode configuration
            if content_type.value not in self.config.content_types:
                raise ValueError(f"Unsupported content type: {content_type}")
            if mode.value not in self.config.learning_modes:
                raise ValueError(f"Unsupported learning mode: {mode}")

            # Build learning prompt
            prompt = self._build_learning_prompt(content, content_type, mode)
            
            # Store learning content
            importance = self._calculate_importance(content_type, mode)
            memory_id = self.memory_manager.store_memory(
                category=content_type.value,
                content={
                    'content': content,
                    'mode': mode.value,
                    'prompt': prompt
                },
                retention_period=retention_period,
                importance=importance
            )
            
            # Update knowledge graph
            self._update_knowledge_graph(content, content_type, mode, memory_id)
            
            # Update learning state
            self._update_learning_state(content_type, mode)
            
            return True
        except Exception as e:
            print(f"Error during learning process: {str(e)}")
            return False

    def _calculate_importance(self, content_type: ContentType, mode: LearningMode) -> float:
        """Calculate content importance"""
        base_importance = 0.5
        
        # Adjust based on content type
        type_weights = {
            ContentType.CODE: 0.8,
            ContentType.TEXT: 0.7,
            ContentType.IMAGE: 0.6,
            ContentType.AUDIO: 0.6,
            ContentType.VIDEO: 0.7
        }
        
        # Adjust based on learning mode
        mode_weights = {
            LearningMode.SUPERVISED: 0.8,
            LearningMode.REINFORCEMENT: 0.7,
            LearningMode.TRANSFER: 0.6,
            LearningMode.UNSUPERVISED: 0.5
        }
        
        return min(1.0, base_importance * type_weights[content_type] * mode_weights[mode])

    def _build_learning_prompt(self, content: Any, content_type: ContentType, mode: LearningMode) -> str:
        """Build learning prompt"""
        content_config = self.config.content_types[content_type.value]
        mode_config = self.config.learning_modes[mode.value]
        
        prompt_parts = [
            f"Content Type: {content_type.value}",
            f"Learning Mode: {mode.value}",
            f"Processing Steps: {content_config['processing']}",
            f"Analysis Methods: {content_config.get('analysis', [])}",
            f"Evaluation Metrics: {mode_config.get('evaluation_metrics', [])}"
        ]
        
        return "\n".join(prompt_parts)

    def _update_learning_state(self, content_type: ContentType, mode: LearningMode):
        """Update learning state"""
        # Update performance metrics
        self.learning_state.performance_metrics["accuracy"] += 0.1
        self.learning_state.performance_metrics["efficiency"] += 0.05
        self.learning_state.performance_metrics["adaptability"] += 0.08
        
        # Store performance metrics
        for metric, value in self.learning_state.performance_metrics.items():
            self.memory_manager.store_performance_metric(metric, value)
        
        # Update knowledge state
        for category, state in self.learning_state.knowledge_state.items():
            self.memory_manager.update_knowledge_state(category, state)
        
        # Record optimization history
        self.learning_state.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "content_type": content_type.value,
            "learning_mode": mode.value,
            "metrics": self.learning_state.performance_metrics.copy()
        })
        
        # Try to trigger knowledge evolution
        self.schedule_evolution()
        
        # Update last updated time
        self.learning_state.last_updated = datetime.now()

    def get_learning_state(self) -> Dict[str, Any]:
        """Get current learning state"""
        return {
            "performance_metrics": self.learning_state.performance_metrics,
            "knowledge_state": self.learning_state.knowledge_state,
            "resource_usage": self.learning_state.resource_usage,
            "last_updated": self.learning_state.last_updated.isoformat()
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return self.learning_state.performance_metrics.copy()

    def cleanup_expired_memories(self) -> int:
        """Clean up expired memories"""
        return self.memory_manager.cleanup_expired_memories()

    def retrieve_memories(self, content_type: ContentType, 
                         max_age: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Retrieve memories of specific type"""
        return self.memory_manager.retrieve_memory(content_type.value, max_age)

    def optimize_memory_storage(self, age_days: int = 30, 
                              min_importance: float = 0.3) -> List[Dict[str, Any]]:
        """Optimize memory storage"""
        archives = self.memory_manager.optimize_storage(
            age_threshold=timedelta(days=age_days),
            min_importance=min_importance
        )
        
        # Record optimization results
        self.learning_state.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "memory_optimization",
            "archives": [{
                "category": a["category"],
                "summary": a["summary"]
            } for a in archives]
        })
        
        return archives

    def restore_archived_memories(self, archive_path: str) -> List[int]:
        """Restore memories from archive"""
        return self.memory_manager.restore_from_archive(archive_path)

    def get_archive_summary(self, archive_path: str) -> Dict[str, Any]:
        """Get archive summary"""
        return self.memory_manager.get_archive_summary(archive_path)

    def _update_knowledge_graph(self, content: Any, content_type: ContentType, 
                              mode: LearningMode, memory_id: int):
        """Update knowledge graph"""
        # Create content node
        content_node_id = f"content_{memory_id}"
        self.knowledge_graph.add_node(
            content_node_id,
            "content",
            {
                'type': content_type.value,
                'mode': mode.value,
                'memory_id': memory_id
            }
        )
        
        # Add related nodes and relationships based on content type
        if content_type == ContentType.TEXT:
            self._process_text_content(content, content_node_id)
        elif content_type == ContentType.CODE:
            self._process_code_content(content, content_node_id)
        elif content_type == ContentType.IMAGE:
            self._process_image_content(content, content_node_id)
        # ... Process other content types ...
        
        # Save graph
        self.knowledge_graph.save_graph()

    def _process_text_content(self, content: str, content_node_id: str):
        """Process text content"""
        # Here you can add text analysis, keyword extraction, etc.
        # Example: Add keyword nodes
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            keyword_node_id = f"keyword_{keyword}"
            self.knowledge_graph.add_node(
                keyword_node_id,
                "keyword",
                {'value': keyword}
            )
            self.knowledge_graph.add_edge(
                content_node_id,
                keyword_node_id,
                "contains_keyword",
                {}
            )

    def _process_code_content(self, content: str, content_node_id: str):
        """Process code content"""
        # Here you can add code analysis, function extraction, etc.
        # Example: Add function nodes
        functions = self._extract_functions(content)
        for func_name, func_info in functions.items():
            func_node_id = f"function_{func_name}"
            self.knowledge_graph.add_node(
                func_node_id,
                "function",
                func_info
            )
            self.knowledge_graph.add_edge(
                content_node_id,
                func_node_id,
                "contains_function",
                {}
            )

    def _process_image_content(self, content: Any, content_node_id: str):
        """Process image content"""
        # Here you can add image analysis, object detection, etc.
        # Example: Add detected object nodes
        objects = self._detect_objects(content)
        for obj_name, obj_info in objects.items():
            obj_node_id = f"object_{obj_name}"
            self.knowledge_graph.add_node(
                obj_node_id,
                "object",
                obj_info
            )
            self.knowledge_graph.add_edge(
                content_node_id,
                obj_node_id,
                "contains_object",
                {}
            )

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords"""
        # Here you can add keyword extraction logic
        # Example implementation
        return set(text.split())

    def _extract_functions(self, code: str) -> Dict[str, Dict[str, Any]]:
        """Extract function information"""
        # Here you can add code analysis logic
        # Example implementation
        return {}

    def _detect_objects(self, image: Any) -> Dict[str, Dict[str, Any]]:
        """Detect objects in image"""
        # Here you can add object detection logic
        # Example implementation
        return {}

    def analyze_knowledge(self) -> Dict[str, Any]:
        """Analyze knowledge state"""
        return self.knowledge_graph.analyze_graph()

    def find_knowledge_path(self, start_id: str, end_id: str, 
                          relationship_types: Optional[Set[str]] = None) -> List[str]:
        """Find knowledge path"""
        return self.knowledge_graph.find_path(start_id, end_id, relationship_types)

    def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> nx.DiGraph:
        """Get related knowledge"""
        return self.knowledge_graph.get_subgraph(node_id, max_depth)

    def evolve_knowledge(self, evolution_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Execute knowledge evolution"""
        evolution_results = self.knowledge_evolution.evolve_knowledge(evolution_threshold)
        
        # Update learning state
        self.learning_state.evolution_history.extend(evolution_results)
        self.learning_state.last_updated = datetime.now()
        
        # Update knowledge state
        evolution_metrics = self.knowledge_evolution.get_evolution_metrics()
        self.learning_state.knowledge_state['evolution_metrics'] = evolution_metrics
        
        return evolution_results

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history"""
        return self.learning_state.evolution_history

    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get evolution metrics"""
        return self.knowledge_evolution.get_evolution_metrics()

    def schedule_evolution(self, interval_hours: int = 24):
        """Schedule periodic evolution"""
        last_evolution = None
        if self.learning_state.evolution_history:
            last_evolution = datetime.fromisoformat(
                self.learning_state.evolution_history[-1].get('timestamp', '')
            )
        
        if (not last_evolution or 
            (datetime.now() - last_evolution) > timedelta(hours=interval_hours)):
            return self.evolve_knowledge()
        
        return [] 